from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .buffers import EpisodeBuffer, FeedbackBuffer


@dataclass
class BaseScheduleState(ABC):
    num_envs: int
    num_timesteps: int
    total_timesteps: int
    training_progress: float
    progress_remaining: float


@dataclass
class PrefScheduleState(BaseScheduleState):
    buffer: EpisodeBuffer
    feed_buffer: FeedbackBuffer
    synth_buffer: FeedbackBuffer


@dataclass
class PrefPPOScheduleState(PrefScheduleState):
    reward_model: nn.Module


class BaseSchedule(ABC):
    @abstractmethod
    def __call__(self, progress_remaining: float, state: BaseScheduleState | None = None) -> Any:
        pass


class ConstantSchedule(BaseSchedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, progress_remaining, state=None):
        return self.value


class LinearSchedule(BaseSchedule):
    def __init__(self, start: float, end: float = 0):
        self.start = start
        self.end = end

    def __call__(self, progress_remaining, state=None):
        return self.end + progress_remaining * (self.start - self.end)


class ExponentialSchedule(BaseSchedule):
    def __init__(self, start: float, end: float = 0, decay: float = 0.5):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, progress_remaining, state=None):
        return self.end + (self.start - self.end) * (self.decay ** (1 - progress_remaining))


class PiecewiseConstantSchedule(BaseSchedule):
    def __init__(self, pieces: list[tuple[int, int]]):
        self.pieces = sorted(pieces, key=lambda p: p[0], reverse=True)

    def __call__(self, progress_remaining, state=None):
        for step, value in self.pieces:
            if 1 - progress_remaining >= step:
                return value
