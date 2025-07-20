from collections import deque
from typing import List

import numpy as np
import torch


class EpisodeBuffer:
    def __init__(self, n_envs, n_episodes, keep_all_eps=False):
        self.n_episodes = n_episodes
        self.keep_all_eps = keep_all_eps
        self.done_eps = deque(maxlen=self.n_episodes if not self.keep_all_eps else None)

        self._running_eps: List = [None] * n_envs

    def add(self, value: torch.Tensor, done: np.ndarray, timesteps: int):
        for env_idx, env_value in enumerate(value):
            if self._running_eps[env_idx] is None:
                self._running_eps[env_idx] = (timesteps, [env_value])
            else:
                age, steps = self._running_eps[env_idx]
                steps.append(env_value)
                self._running_eps[env_idx] = (age, steps)

        for env_idx in np.argwhere(done).reshape(-1):
            age, steps = self._running_eps[env_idx]
            self.done_eps.append((age, torch.stack(steps)))
            self._running_eps[env_idx] = None

    def get_episodes(self):
        running = [torch.stack(ep[1]) for ep in self._running_eps if ep is not None]
        done = [ep[1] for ep in self.done_eps]
        if self.keep_all_eps:
            done = done[-self.n_episodes:]
        return done + running

    def get_episode_ages(self):
        running_ages = [ep[0] for ep in self._running_eps if ep is not None]
        done_ages = [ep[0] for ep in self.done_eps]
        if self.keep_all_eps:
            done_ages = done_ages[-self.n_episodes:]
        return done_ages + running_ages