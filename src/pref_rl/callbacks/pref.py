from abc import ABC, abstractmethod
import itertools
import torch
from .reward_mod import RewardModifierCallback
from ..utils.pref import EpisodeBuffer, Teacher, Sampler


class BasePrefCallback(RewardModifierCallback, ABC):
    def __init__(self,
        device: str = 'cpu',
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
        on_first_trained: callable = None,
        margins_stats_window_size: int = 100,
        on_trained: callable = None,
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)

        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.sampling_method = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.train_teacher = teacher
        self.teacher_kwargs = teacher_kwargs
        self.n_steps_first_train = n_steps_first_train
        self.on_first_trained = on_first_trained
        self.margins_stats_window_size = margins_stats_window_size
        self.on_trained = on_trained

        assert self.ann_buffer_size_eps is None or self.margins_stats_window_size <= self.ann_buffer_size_eps

        self.steps_since_train = 0
        self.has_trained = False
        self.num_feed = 0
        self.keep_training = None

        self.device = torch.device(
            device or
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )


    def _init_callback(self):
        obs_size, act_size = self._get_input_sizes()
        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size)
        self.train_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher=self.train_teacher, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.segment_buffer = torch.empty((self.max_feed, 2, self.segment_size, segment_dim), device=self.device).detach()
        self.preference_buffer = torch.empty((self.max_feed,), device=self.device).detach()


    @abstractmethod
    def _get_predictor(self):
        raise NotImplementedError


    def _expand_data(self, sampling_method: str):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.episodes, num_samples, sampling_method, self._get_predictor())

        preferences, keep_indices = self.train_teacher.query_segments(rewards.detach())

        start, end = self.num_feed, self.num_feed + len(keep_indices)
        self.segment_buffer[start:end] = state_actions[keep_indices].detach().to(self.device)
        self.preference_buffer[start:end] = preferences.detach().to(self.device)

        self.num_feed += len(keep_indices)
        self.logger.record('pref/num_feed', self.num_feed)


    @abstractmethod
    def _train_predictor(self):
        raise NotImplementedError


    def _on_step(self):
        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        if self.locals['dones'].any():
            recent_eps = list(itertools.islice(reversed(self.buffer.episodes), self.margins_stats_window_size))
            self.train_teacher.update_thresholds(recent_eps)

        buffer_empty = len(self.buffer.episodes) == 0
        if self.n_steps_first_train is None and not buffer_empty:
            self.n_steps_first_train = self.num_timesteps

        should_first_train = not self.has_trained and self.n_steps_first_train is not None and self.steps_since_train > self.n_steps_first_train
        should_train = self.steps_since_train > self.n_steps_reward
        feedback_left = self.num_feed < self.max_feed

        if (should_first_train or should_train) and (feedback_left or self.keep_training):
            sampling_method = 'uniform' if not self.has_trained else self.sampling_method
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
        else:
            self.steps_since_train += self.training_env.num_envs

        return True if not self.has_trained else super()._on_step()
