from abc import ABC, abstractmethod
from typing import Callable

from .schedules import BaseSchedule, ConstantSchedule, PrefScheduleState, BaseScheduleState


class TrainingSchedule(BaseSchedule, ABC):
    @abstractmethod
    def __call__(self, progress_remaining: float, state: BaseScheduleState | None = None) -> tuple[bool, int]:
        pass


class BasePrefSchedule(TrainingSchedule):
    """
    A stateful schedule that determines when to train and how many feedback pairs to collect.
    
    Returns the number of feedback pairs to query on each step. Returns 0 when no training should occur.
    """

    def __init__(self,
                 n_steps_reward: int = 32_000,
                 max_feed: int | None = 2_000,
                 feed_batch_size: int | None = None,
                 batch_size_schedule: Callable | None = None,
                 n_steps_first_train: int | None = None,
                 n_steps_last_train: int | None = None,
                 keep_training: bool = False):
        """
        :param n_steps_reward: Steps between reward model training
        :param n_steps_first_train: Steps before first training (None = auto-detect from episodes)
        :param n_steps_last_train: Steps after which to stop training
        :param max_feed: Maximum total feedback pairs to collect
        :param batch_size_schedule: Schedule for determining batch size (overrides feed_batch_size)
        :param feed_batch_size: Fixed batch size for feedback collection
        """
        if not feed_batch_size and not batch_size_schedule:
            raise ValueError('Either feed_batch_size or batch_size_schedule must be set')
        
        self.n_steps_reward = n_steps_reward
        self.n_steps_first_train = n_steps_first_train
        self.n_steps_last_train = n_steps_last_train
        self.max_feed = max_feed
        self.batch_size_schedule = batch_size_schedule or ConstantSchedule(feed_batch_size)
        self.keep_training = keep_training


    def _is_training_checkpoint(self, num_timesteps: int, num_envs: int) -> bool:
        """Determine if training should occur at this timestep."""
        if self.n_steps_first_train is None:
            return False

        step_remainder = (num_timesteps - self.n_steps_first_train) % self.n_steps_reward
        should_train = num_timesteps >= self.n_steps_first_train and step_remainder < num_envs

        should_stop_training = self.n_steps_last_train is not None and num_timesteps >= self.n_steps_last_train

        return should_train and not should_stop_training


    def __call__(self, progress_remaining: float, state: PrefScheduleState | None = None) -> int:
        """
        Return the number of feedback pairs to collect at this step.
        
        :param progress_remaining: Progress remaining (1.0 to 0.0)
        :param state: Schedule state containing buffer and other info
        :return: Number of feedback pairs to collect (0 = no training)
        """ 
        buffer_has_done_eps = len(state.buffer.done_eps) > 0
        if self.n_steps_first_train is None and buffer_has_done_eps:
            self.n_steps_first_train = state.num_timesteps

        feedback_left = self.max_feed is None or state.feed_buffer.position < self.max_feed
        is_checkpoint = self._is_training_checkpoint(state.num_timesteps, state.num_envs)
        should_train = is_checkpoint and (feedback_left or self.keep_training)

        num_samples = int(self.batch_size_schedule(state.progress_remaining, state)) if should_train else 0
        return should_train, num_samples
