from abc import ABC, abstractmethod
import itertools
from typing import Callable

import torch
from torch.utils.data import Dataset, ConcatDataset

from .reward_mod import RewardModifierCallback
from ..utils.buffers import EpisodeBuffer, FeedbackBuffer
from ..utils.data import MaskedDataset
from ..utils.sampler import DisagreementMetric, EntropyMetric, Sampler
from ..utils.schedules import PrefScheduleState
from ..utils.synthetic import BaseSynthesizer
from ..utils.train_schedules import TrainingSchedule
from ..utils.teacher import Teacher


class BasePrefCallback(RewardModifierCallback, ABC):
    def __init__(self,
        schedule: TrainingSchedule,
        segment_size: int = 50,
        ann_buffer_size_eps: int | None = None,
        margins_stats_window_size: int = 100,
        sampler: str = 'uniform',
        sampler_kwargs: dict = {},
        sample_uniform_on_first_train: bool = True,  # For consistency with B-Pref
        log_sampler_metrics: bool = True,
        teacher: str = 'oracle',
        teacher_kwargs: dict = {},
        feed_buffer_size: int | None = None,
        synth_buffer_size: int | None = None,
        synth_schedule: TrainingSchedule | None = None,
        synthesizer: BaseSynthesizer | None = None,
        on_first_trained: Callable | None = None,
        on_trained: Callable | None = None,
        keep_all_eps: bool = False,
        save_episode_data: bool = False,
        log_prefix='pref/',
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)

        self.schedule = schedule
        self.segment_size = segment_size
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.margins_stats_window_size = margins_stats_window_size
        self.sampler_metric = sampler
        self.sampler_kwargs = sampler_kwargs
        self.sample_uniform_on_first_train = sample_uniform_on_first_train
        self.log_sampler_metrics = log_sampler_metrics
        self.train_teacher_kind = teacher
        self.teacher_kwargs = teacher_kwargs
        self.feed_buf_size = feed_buffer_size or self.schedule.max_feed
        self.synth_schedule = synth_schedule
        self.synth_buffer_size = synth_buffer_size or 0
        self.on_first_trained = on_first_trained
        self.on_trained = on_trained
        self.keep_all_eps = keep_all_eps or save_episode_data
        self.uniform_frac = self.sampler_kwargs.pop('uniform_fraction', 0.0)
        self.synthesizer = synthesizer

        if self.synth_schedule and not synth_buffer_size:
            raise ValueError('synth_buffer_size must be provided if synth_ratio is set')
        if self.ann_buffer_size_eps and self.margins_stats_window_size > self.ann_buffer_size_eps:
            raise ValueError('margin_stats_window_size must be less than or equal to ann_buffer_size_eps')

        self.has_trained = False
        self.training_progress = 0.0
        self.n_steps_train_end = None

        self.device = torch.device(
            device or
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    @property
    def num_feed(self):
        return self.feed_buffer.position


    def _init_callback(self):
        if isinstance(self.sampler_metric, str):
            metric_classes = {
                'disagreement': DisagreementMetric,
                'entropy': EntropyMetric,
            }
            metric_class = metric_classes.get(self.sampler_metric)
            self.sampler_metric = metric_class() if metric_class else None

        obs_size, act_size = self._get_input_sizes()
        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps, keep_all_eps=self.keep_all_eps)
        self.sampler = Sampler(self.segment_size, obs_size, act_size, self.sampler_metric, **self.sampler_kwargs)
        self.uniform_sampler = Sampler(self.segment_size, obs_size, act_size, **self.sampler_kwargs)
        self.train_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher=self.train_teacher_kind, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.feed_buffer = FeedbackBuffer(self.feed_buf_size, self.segment_size, segment_dim, self.device)
        self.synth_buffer = FeedbackBuffer(self.synth_buffer_size, self.segment_size, segment_dim, self.device)

        self.total_timesteps = self.model._total_timesteps


    def _create_schedule_state(self):
        progress_remaining = 1.0 - float(self.num_timesteps) / float(self.total_timesteps)
        return PrefScheduleState(
            num_timesteps=self.num_timesteps,
            total_timesteps=self.total_timesteps,
            training_progress=self.training_progress,
            progress_remaining=progress_remaining,
            buffer=self.buffer,
            feed_buffer=self.feed_buffer,
            synth_buffer=self.synth_buffer,
            num_envs=self.training_env.num_envs,
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
        state_actions, rewards, metrics, segment_metas = sampler.sample_pairs(episodes, episode_ages, num_specific, reward_model=reward_model, schedule_state=schedule_state, log_metrics=self.log_sampler_metrics)

        if num_uniform > 0:
            state_actions_uniform, rewards_uniform, _, segment_metas_uniform = sampler.sample_pairs(episodes, episode_ages, num_uniform, reward_model=reward_model, schedule_state=schedule_state, log_metrics=False)
            state_actions = torch.cat([state_actions, state_actions_uniform], dim=0)
            rewards = torch.cat([rewards, rewards_uniform], dim=1)
            segment_metas.extend(segment_metas_uniform)

        return state_actions, rewards, metrics, segment_metas


    def _expand_real_data(self, sampler: Sampler, num_samples: int):
        episodes = self.buffer.get_episodes()
        episode_metas = self.buffer.get_episode_metas()
        reward_model = self._get_predictor()
        state_actions, rewards, sampler_metrics, segment_metas = self._sample_segments(sampler, episodes, episode_metas, num_samples, reward_model)

        if self.log_sampler_metrics and sampler_metrics:
            self._log_metrics_stats(sampler_metrics, prefix='sampler/')

        preferences, keep_indices = self.train_teacher.query_segments(rewards.detach())
        weights = torch.ones_like(preferences[keep_indices])
        selected_segment_metas = [segment_metas[idx] for idx in keep_indices]
        num_added = self.feed_buffer.add(state_actions[keep_indices], preferences[keep_indices], weights, selected_segment_metas)

        if self.num_feed == self.schedule.max_feed:
            self.n_steps_train_end = self.num_timesteps

        self._update_training_progress()
        
        metrics_to_log = {
            'pref/feed_buffer_pos': self.feed_buffer.position,
            'pref/feed_buffer_size': self.feed_buffer.size,
        }
        self.logger.record_with_progress(metrics_to_log, self.num_feed, self.training_progress)

        return num_added


    def _expand_synth_data(self, num_samples: int):
        episodes = self.buffer.get_episodes(return_all=True)
        episode_metas = self.buffer.get_episode_metas(return_all=True)
        schedule_state = self._create_schedule_state()
        segments, preferences, metrics, weights = self.synthesizer.generate_pairs(episodes, episode_metas, num_samples, schedule_state)
        self._log_metrics_stats(metrics, prefix='synth/')

        num_added = self.synth_buffer.add(segments, preferences, weights)

        synth_metrics = {
            'pref/num_synth': self.synth_buffer.position,
            'pref/synth_buffer_pos': self.synth_buffer.position,
            'pref/synth_buffer_size': self.synth_buffer.size,
        }
        self.logger.record_with_progress(synth_metrics, self.num_feed, self.training_progress)

        return num_added


    def _update_training_progress(self):
        if self.n_steps_train_end:
            self.training_progress = (self.num_timesteps - self.schedule.n_steps_first_train) / self.n_steps_train_end
        else:
            self.training_progress = self.num_feed / self.schedule.max_feed


    @abstractmethod
    def _train_predictor(self, dataset: Dataset):
        raise NotImplementedError


    def _add_steps_to_buffer(self):
        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        meta = {}
        self.buffer.add(annotations, self.locals['dones'], meta=meta, timesteps=self.num_timesteps)


    def _on_episode_done(self):
        # Update teacher thresholds
        recent_eps = list(itertools.islice(reversed(self.buffer.get_episodes()), self.margins_stats_window_size))
        self.train_teacher.update_thresholds(recent_eps)

        # Log episode metrics
        episode_metrics = {'pref/num_episodes': len(self.buffer.done_eps)}
        self.logger.record_with_progress(episode_metrics, self.num_feed, self.training_progress)


    def _handle_train(self, dataset: Dataset):
        self._train_predictor(dataset)

        if self.on_trained:
            self.on_trained()

        if not self.has_trained:
            self.has_trained = True
            if self.on_first_trained: self.on_first_trained()

        self.logger.dump(step=self.num_timesteps)


    def _get_sampler(self):
        if self.sample_uniform_on_first_train and not self.has_trained:
            return self.uniform_sampler
        else:
            return self.sampler


    def _get_dataset(self, should_train_real: bool, should_train_synth: bool):
        datasets = []
        if should_train_real:
            datasets.append(MaskedDataset(self.feed_buffer.get_dataset(), 0))
        if should_train_synth:
            datasets.append(MaskedDataset(self.synth_buffer.get_dataset(), 1))
        return ConcatDataset(datasets)


    def _on_step(self):
        self._add_steps_to_buffer()

        if self.locals['dones'].any():
            self._on_episode_done()

        schedule_state = self._create_schedule_state()
        should_train_real, num_samples_real = self.schedule(schedule_state.progress_remaining, schedule_state)
        if num_samples_real > 0:
            self._expand_real_data(self._get_sampler(), num_samples_real)

        should_train_synth, num_samples_synth = self.synth_schedule(schedule_state.progress_remaining, schedule_state) if self.synth_schedule else (False, 0)
        if num_samples_synth > 0:
            self._expand_synth_data(num_samples_synth)

        if should_train_real or should_train_synth:
            dataset = self._get_dataset(should_train_real, should_train_synth)
            if not should_train_real:
                train_acc_threshold_reward = self.train_acc_threshold_reward
                self.train_acc_threshold_reward = 0
            self._handle_train(dataset)
            if not should_train_real:
                self.train_acc_threshold_reward = train_acc_threshold_reward

        return super()._on_step()
