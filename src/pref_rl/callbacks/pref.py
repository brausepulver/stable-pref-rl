from abc import ABC, abstractmethod
import itertools
from typing import Callable, Optional

import torch

from .reward_mod import RewardModifierCallback
from ..utils.buffers import EpisodeBuffer, FeedbackBuffer
from ..utils.sampler import DisagreementMetric, EntropyMetric, Sampler
from ..utils.schedules import ConstantSchedule, PrefScheduleState
from ..utils.synthetic import TemporalSynthesizer
from ..utils.teacher import Teacher


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
        synth_ratio: Optional[float] = None,
        synth_start_step: Optional[int] = None,
        synth_stop_step: Optional[int] = None,
        synth_buffer_size: int | None = None,
        synth_teacher_kwargs: Optional[dict] = None,
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
        self.synth_ratio = synth_ratio or 0
        self.synth_enabled = synth_ratio and synth_ratio > 0
        self.synth_start_step = synth_start_step or 0
        self.synth_stop_step = synth_stop_step or float('inf')
        self.synth_buffer_size = synth_buffer_size
        self.synth_teacher_kwargs = synth_teacher_kwargs or {}

        if self.synth_enabled and not synth_buffer_size:
            raise ValueError('synth_buffer_size must be provided if synth_ratio is set')

        self.uniform_frac = self.sampler_kwargs.get('uniform_fraction', 0.0) if self.sampling_method != 'uniform' else 0.0

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
        self.num_synth = 0

        self.device = torch.device(
            device or
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    def _init_callback(self):
        sampling_metrics = {
            'disagreement': DisagreementMetric,
            'entropy': EntropyMetric,
        }
        sampling_metric_cls = sampling_metrics.get(self.sampling_method, None)
        sampling_metric = sampling_metric_cls() if sampling_metric_cls else None

        obs_size, act_size = self._get_input_sizes()
        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps, keep_all_eps=self.save_episode_data)
        self.sampler = Sampler(self.segment_size, obs_size, act_size, sampling_metric)
        self.uniform_sampler = Sampler(self.segment_size, obs_size, act_size)
        self.train_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher=self.train_teacher_str, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.feed_buffer = FeedbackBuffer(self.feed_buf_size, self.segment_size, segment_dim, self.device)

        self.synthesizer = TemporalSynthesizer(self.segment_size, obs_size, act_size, **self.synth_teacher_kwargs)
        self.synth_buffer = FeedbackBuffer(self.synth_buffer_size, self.segment_size, segment_dim, self.device)

        self.total_timesteps = self.model._total_timesteps


    def _create_schedule_state(self):
        progress_remaining = 1.0 - float(self.num_timesteps) / float(self.total_timesteps)
        return PrefScheduleState(
            num_timesteps=self.num_timesteps,
            total_timesteps=self.total_timesteps,
            training_progress=self.training_progress,
            progress_remaining=progress_remaining,
            buffer=self.buffer
        )


    def _log_metrics_stats(self, metrics: dict, prefix: str = ''):
        metrics_stats = {}

        for metric_name, metric_values in metrics.items():
            metrics_stats[f'{metric_name}_mean'] = metric_values.mean().cpu().item()
            metrics_stats[f'{metric_name}_std'] = metric_values.std().cpu().item()

        self.logger.record_with_progress(metrics_stats, self.num_feed, self.training_progress, prefix=f'reward_model/{prefix}')


    def _sample_segments(self, sampler: Sampler, episodes, episode_ages, num_samples, reward_model):
        num_uniform = int(num_samples * self.uniform_frac)
        num_specific = num_samples - num_uniform

        schedule_state = self._create_schedule_state()
        state_actions, rewards, _ = sampler.sample_pairs(episodes, episode_ages, num_specific, reward_model=reward_model, schedule_state=schedule_state)

        if num_uniform > 0:
            state_actions_uniform, rewards_uniform, _ = sampler.sample_pairs(episodes, episode_ages, num_uniform, reward_model=reward_model, schedule_state=schedule_state)
            state_actions = torch.cat([state_actions, state_actions_uniform], dim=0)
            rewards = torch.cat([rewards, rewards_uniform], dim=1)

        metrics = sampler.compute_logging_metrics(state_actions, reward_model, schedule_state=schedule_state) if self.log_sampler_metrics else {}
        return state_actions, rewards, metrics


    def _expand_real_data(self, sampler: Sampler, feed_batch_size: int):
        episodes = self.buffer.get_episodes()
        episode_ages = self.buffer.get_episode_ages()
        num_samples = min(feed_batch_size, self.max_feed - self.num_feed)
        reward_model = self._get_predictor()

        state_actions, rewards, sampler_metrics = self._sample_segments(sampler, episodes, episode_ages,num_samples, reward_model)

        if self.log_sampler_metrics and sampler_metrics:
            self._log_metrics_stats(sampler_metrics, prefix='sampler/')

        preferences, keep_indices = self.train_teacher.query_segments(rewards.detach())
        weights = torch.ones_like(preferences[keep_indices])
        num_added = self.feed_buffer.add(state_actions[keep_indices], preferences[keep_indices], weights)
        return num_added


    def _expand_synth_data(self, feed_batch_size: int):
        episodes = self.buffer.get_episodes()
        episode_ages = self.buffer.get_episode_ages()
        num_samples = int(feed_batch_size * self.synth_ratio)

        segments, preferences, metrics, weights = self.synthesizer.generate_pairs(episodes, episode_ages, num_samples, self.num_timesteps)
        self._log_metrics_stats(metrics, prefix='synth/')
        
        num_added = self.synth_buffer.add(segments, preferences, weights)
        return num_added


    def _expand_data(self, sampler: Sampler):
        schedule_state = self._create_schedule_state()
        feed_batch_size = int(self.feed_schedule(schedule_state.progress_remaining, schedule_state))

        num_added = self._expand_real_data(sampler, feed_batch_size)
        self.num_feed += num_added
        self.training_progress = self.num_feed / self.max_feed
        
        metrics_to_log = {
            'pref/num_feed': self.num_feed,
            'pref/training_progress': self.training_progress,
            'pref/feed_buffer_pos': self.feed_buffer.position,
            'pref/feed_buffer_size': self.feed_buffer.size,
        }
        self.logger.record_with_progress(metrics_to_log, self.num_feed, self.training_progress)

        should_collect_synth = self.synth_start_step <= self.num_timesteps <= self.synth_stop_step
        if self.synth_enabled and should_collect_synth:
            self.num_synth += self._expand_synth_data(feed_batch_size)
            synth_metrics = {
                'pref/num_synth': self.num_synth,
                'pref/synth_buffer_pos': self.synth_buffer.position,
                'pref/synth_buffer_size': self.synth_buffer.size,
            }
            self.logger.record_with_progress(synth_metrics, self.num_feed, self.training_progress)


    @abstractmethod
    def _train_predictor(self):
        raise NotImplementedError


    def _should_train(self) -> bool:
        should_start_training = not self.has_trained and self.n_steps_first_train is not None and self.steps_since_train >= self.n_steps_first_train
        should_train = self.has_trained and self.steps_since_train >= self.n_steps_reward
        should_stop_training = self.n_steps_last_train is not None and self.num_timesteps >= self.n_steps_last_train
        return (should_start_training or should_train) and not should_stop_training


    def _on_step(self):
        self.steps_since_train += self.training_env.num_envs

        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'], self.num_timesteps)

        if self.locals['dones'].any():
            recent_eps = list(itertools.islice(reversed(self.buffer.get_episodes()), self.margins_stats_window_size))
            self.train_teacher.update_thresholds(recent_eps)

            episode_metrics = {'pref/num_episodes': len(self.buffer.done_eps)}
            self.logger.record_with_progress(episode_metrics, self.num_feed, self.training_progress)

        buffer_has_done = len(self.buffer.done_eps) > 0
        feedback_left = self.num_feed < self.max_feed

        if self.n_steps_first_train is None and buffer_has_done:
            self.n_steps_first_train = self.num_timesteps

        if self._should_train() and (feedback_left or self.keep_training):
            sampler = self.sampler if self.has_trained else self.uniform_sampler
            if feedback_left:
                self._expand_data(sampler)
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

        return super()._on_step()
