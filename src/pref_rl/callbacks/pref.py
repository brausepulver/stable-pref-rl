from abc import ABC, abstractmethod
import gymnasium as gym
import torch
from .reward_mod import RewardModifierCallback
from ..utils.pref import EpisodeBuffer, Teacher, Sampler


class BasePrefCallback(RewardModifierCallback, ABC):
    def __init__(self,
        n_steps_reward: int = 32_000,
        ann_buffer_size_eps: int = 100,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_batch_size: int = 200,
        teacher: str = None,
        teacher_kwargs: dict = {},
        log_prefix='pref/',
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
        self.num_feed = 0


    def _init_callback(self):
        super()._init_callback()

        self.observation_size = self.training_env.observation_space.shape[0]
        self.action_size = 1 if isinstance(self.training_env.action_space, gym.spaces.Discrete) else self.training_env.action_space.shape[0]

        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=self.observation_size, action_size=self.action_size)
        self.teacher = Teacher(segment_size=self.segment_size, teacher=self.teacher, teacher_kwargs=self.teacher_kwargs)

        segment_dim = self.observation_size + self.action_size
        self.segment_buffer = torch.empty((0, 2, self.segment_size, segment_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    def _expand_data(self):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.episodes, num_samples, self.sampling_method, self._get_predictor())

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

        checkpoint_reached = self.num_timesteps % self.n_steps_reward == 0
        feedback_left = self.num_feed < self.max_feed
        buffer_empty = len(self.buffer.episodes) == 0

        if checkpoint_reached and feedback_left and not buffer_empty:
            self._expand_data()
            self._train_predictor()
            self._on_event()

        return super()._on_step()
