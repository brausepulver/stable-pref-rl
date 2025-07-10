from abc import ABC, abstractmethod
import itertools
from typing import Callable

import torch

from .reward_mod import RewardModifierCallback
from ..utils.pref import EpisodeBuffer, Teacher, Sampler
from ..config import ConstantSchedule


class BasePrefCallback(RewardModifierCallback, ABC):
    def __init__(self,
        device: str = 'cpu',
        n_steps_reward: int = 32_000,
        ann_buffer_size_eps: int | None = None,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_buffer_size: int | None = None,
        feed_batch_size: int | None = 200,
        feed_schedule: Callable | None = None,
        teacher: str = 'oracle',
        teacher_kwargs: dict = {},
        log_prefix='pref/',
        n_steps_first_train: int | None = None,
        n_steps_last_train: int | None = None,
        on_first_trained: Callable | None = None,
        margins_stats_window_size: int = 100,
        on_trained: Callable | None = None,
        log_sampler_metrics: bool = True,
        sampler_kwargs: dict = {},
        save_episode_data: bool = False,
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)

        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.sampling_method = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_buf_size = feed_buffer_size or max_feed
        self.train_teacher_str = teacher
        self.teacher_kwargs = teacher_kwargs
        self.n_steps_first_train = n_steps_first_train
        self.n_steps_last_train = n_steps_last_train
        self.on_first_trained = on_first_trained
        self.margins_stats_window_size = margins_stats_window_size
        self.on_trained = on_trained
        self.log_sampler_metrics = log_sampler_metrics
        self.sampler_kwargs = sampler_kwargs
        self.save_episode_data = save_episode_data

        if not feed_schedule and not feed_batch_size:
            raise ValueError('Either feed_batch_size or feed_schedule must be set')
        if feed_batch_size and self.feed_buf_size and self.feed_buf_size < feed_batch_size:
            raise ValueError('feed_buffer_size must be greater than or equal to feed_batch_size')
        self.feed_schedule = feed_schedule or ConstantSchedule(feed_batch_size)

        assert self.ann_buffer_size_eps is None or self.margins_stats_window_size <= self.ann_buffer_size_eps

        self.steps_since_train = 0
        self.has_trained = False
        self.num_feed = 0
        self.training_progress = 0.0
        self.keep_training = None
        self.n_steps_train_total = None
        self.buffer_position = 0

        self.device = torch.device(
            device or
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )

        self.run = None
        try:
            import wandb
            if wandb.run is not None:
                self.run = wandb.run
                self.run.define_metric(step_metric='pref/training_progress', name='pref/*')
                self.run.define_metric(step_metric='pref/num_feed', name='pref/*')
        except ImportError:
            pass


    def _init_callback(self):
        obs_size, act_size = self._get_input_sizes()
        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps, keep_all_eps=self.save_episode_data)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size)
        self.train_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher=self.train_teacher_str, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.segment_buffer = torch.empty((self.feed_buf_size, 2, self.segment_size, segment_dim), device=self.device).detach()
        self.preference_buffer = torch.empty((self.feed_buf_size,), device=self.device).detach()

        self.total_timesteps = self.model._total_timesteps


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    def _sample_segments(self, episodes, num_samples, sampling_method, reward_model):
        uniform_frac = self.sampler_kwargs.get('uniform_fraction', 0.0) if sampling_method != 'uniform' else 0.0
        num_uniform = int(num_samples * uniform_frac)
        num_specific = num_samples - num_uniform

        state_actions, rewards, _ = self.sampler.sample_segments(
            episodes,
            num_specific,
            sampling_method,
            reward_model,
        )

        if num_uniform > 0:
            state_actions_uniform, rewards_uniform, _ = self.sampler.sample_segments(
                episodes,
                num_uniform,
                'uniform',
                reward_model,
            )
            state_actions = torch.cat([state_actions, state_actions_uniform], dim=0)
            rewards = torch.cat([rewards, rewards_uniform], dim=1)

        metrics = self.sampler.compute_metrics(state_actions, reward_model) if self.log_sampler_metrics else {}
        return state_actions, rewards, metrics


    def _add_to_buffer(self, state_actions, preferences, keep_indices):
        num_items = len(keep_indices)
        
        if num_items > 0:
            start_pos = self.buffer_position % self.feed_buf_size
            end_pos = (self.buffer_position + num_items) % self.feed_buf_size
            
            if start_pos < end_pos:
                # No wraparound
                self.segment_buffer[start_pos:end_pos] = state_actions[keep_indices].detach().to(self.device)
                self.preference_buffer[start_pos:end_pos] = preferences.detach().to(self.device)
            else:
                # Wraparound
                first_chunk = self.feed_buf_size - start_pos
                self.segment_buffer[start_pos:] = state_actions[keep_indices[:first_chunk]].detach().to(self.device)
                self.preference_buffer[start_pos:] = preferences[:first_chunk].detach().to(self.device)
                if end_pos > 0:
                    self.segment_buffer[:end_pos] = state_actions[keep_indices[first_chunk:]].detach().to(self.device)
                    self.preference_buffer[:end_pos] = preferences[first_chunk:].detach().to(self.device)
            
            self.buffer_position += num_items


    def _expand_data(self, sampling_method: str):
        progress_remaining = 1.0 - float(self.num_timesteps) / float(self.total_timesteps)
        feed_batch_size = int(self.feed_schedule(
            progress_remaining,
            num_timesteps=self.num_timesteps,
            total_timesteps=self.total_timesteps
        ))

        episodes = self.buffer.get_episodes()
        num_samples = min(feed_batch_size, self.max_feed - self.num_feed)
        reward_model = self._get_predictor()

        state_actions, rewards, sampler_metrics = self._sample_segments(
            episodes,
            num_samples,
            sampling_method,
            reward_model
        )

        if self.log_sampler_metrics and sampler_metrics:
            self._log_sampler_metrics(sampler_metrics, prefix='sampler/')

        preferences, keep_indices = self.train_teacher.query_segments(rewards.detach())

        self._add_to_buffer(state_actions, preferences, keep_indices)

        self.num_feed += len(keep_indices)
        self.training_progress = self.num_feed / self.max_feed
        self.logger.record('pref/num_feed', self.num_feed)
        self.logger.record('pref/feed_buffer_pos', self.buffer_position)
        self.logger.record('pref/training_progress', self.training_progress)


    def _log_sampler_metrics(self, sampler_metrics: dict, prefix: str = ''):
        for metric_name, metric_values in sampler_metrics.items():
            metric_mean = metric_values.mean().cpu().item()
            metric_std = metric_values.std().cpu().item()

            self.logger.record(f'reward_model/{prefix}{metric_name}_mean', metric_mean)
            self.logger.record(f'reward_model/{prefix}{metric_name}_std', metric_std)

            if self.run:
                self.run.log({
                    f'reward_model/{prefix}{metric_name}_mean': metric_mean,
                    f'reward_model/{prefix}{metric_name}_std': metric_std,
                    'pref/num_feed': self.num_feed,
                    'pref/training_progress': self.training_progress,
                })


    @abstractmethod
    def _train_predictor(self):
        raise NotImplementedError


    def _on_step(self):
        self.steps_since_train += self.training_env.num_envs

        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        if self.locals['dones'].any():
            recent_eps = list(itertools.islice(reversed(self.buffer.get_episodes()), self.margins_stats_window_size))
            self.train_teacher.update_thresholds(recent_eps)

            self.logger.record('pref/num_episodes', len(self.buffer.done_eps))

        buffer_has_done = len(self.buffer.done_eps) > 0
        if self.n_steps_first_train is None and buffer_has_done:
            self.n_steps_first_train = self.num_timesteps

        should_first_train = not self.has_trained and self.n_steps_first_train is not None and self.steps_since_train >= self.n_steps_first_train
        should_train = self.has_trained and self.steps_since_train >= self.n_steps_reward
        feedback_left = self.num_feed < self.max_feed
        should_stop_training = self.n_steps_last_train is not None and self.num_timesteps >= self.n_steps_last_train

        if (should_first_train or should_train) and (feedback_left or self.keep_training) and not should_stop_training:
            sampling_method = self.sampling_method if self.has_trained else 'uniform'
            if feedback_left:
                self._expand_data(sampling_method)
            self._train_predictor()
            self.steps_since_train = 0

            if self.on_trained:
                self.on_trained()

            if not self.has_trained:
                self.has_trained = True
                if self.on_first_trained: self.on_first_trained()

            self.logger.dump(step=self.num_timesteps)
            self.n_steps_train_total = self.num_timesteps - self.n_steps_first_train

        elif self.has_trained and not feedback_left:
            self.training_progress = (self.num_timesteps - self.n_steps_first_train) / self.n_steps_train_total

        return True if not self.has_trained else super()._on_step()
