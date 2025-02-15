from abc import ABC, abstractmethod
import gymnasium as gym
import torch
from .reward_mod import RewardModifierCallback
from ..utils.pref import EpisodeBuffer, Teacher, Sampler


class BasePrefCallback(RewardModifierCallback, ABC):
    def __init__(self,
        n_steps_reward: int = 32_000,
        ann_buffer_size_eps: int = None,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_batch_size: int = 200,
        teacher: str = None,
        teacher_kwargs: dict = {},
        log_prefix='pref/',
        n_steps_first_train: int = None,
        on_first_train: callable = None,
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)
        
        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.sampling_method = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.teacher = teacher
        self.teacher_kwargs = teacher_kwargs
        self.n_steps_first_train = n_steps_first_train
        self.on_first_train = on_first_train

        self.has_trained = False
        self.num_feed = 0


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()
        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size)
        self.teacher = Teacher(segment_size=self.segment_size, teacher=self.teacher, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.segment_buffer = torch.empty((0, 2, self.segment_size, segment_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    def _expand_data(self, sampling_method: str):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.episodes, num_samples, sampling_method, self._get_predictor())

        preferences, keep_indices = self.teacher.query_segments(rewards)

        self.segment_buffer = torch.cat([self.segment_buffer, state_actions[keep_indices]])
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])
        
        self.num_feed += len(keep_indices)
        self.logger.record(f'pref/num_feed', self.num_feed)


    @abstractmethod
    def _train_predictor(self):
        raise NotImplementedError


    def _on_step(self):
        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        if self.locals['dones'].any():
            self.teacher.update_thresholds(self.buffer.episodes)

        buffer_empty = len(self.buffer.episodes) == 0
        if not buffer_empty and self.n_steps_first_train is None:
            self.n_steps_first_train = self.num_timesteps

        should_train = self.n_steps_first_train is not None
        checkpoint_reached = should_train and (self.num_timesteps - self.n_steps_first_train) % self.n_steps_reward == 0
        feedback_left = self.num_feed < self.max_feed

        if checkpoint_reached and feedback_left:
            sampling_method = 'uniform' if not self.has_trained else self.sampling_method
            self._expand_data(sampling_method)
            self._train_predictor()
            self._on_event()

            if not self.has_trained:
                self.has_trained = True
                if self.on_first_train: self.on_first_train()

        return True if not self.has_trained else super()._on_step()
