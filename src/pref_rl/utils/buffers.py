from collections import deque
import numpy as np
import torch


class EpisodeBuffer:
    def __init__(self, n_envs, n_episodes, keep_all_eps=False):
        self.n_episodes = n_episodes
        self.keep_all_eps = keep_all_eps
        self.done_eps = deque(maxlen=self.n_episodes if not self.keep_all_eps else None)
        self._running_eps = [[] for _ in range(n_envs)]


    def add(self, value: torch.Tensor, done: np.ndarray):
        for env_idx, env_value in enumerate(value):
            self._running_eps[env_idx].append(env_value)

        for env_idx in np.argwhere(done).reshape(-1):
            episode = self._running_eps[env_idx]
            self.done_eps.append(torch.stack(episode))
            episode.clear()


    def get_episodes(self):
        running_ep_tensors = [torch.stack(ep) for ep in self._running_eps if len(ep) > 0]
        done_eps = list(self.done_eps)
        if self.keep_all_eps:
            done_eps = done_eps[-self.n_episodes:]
        return done_eps + running_ep_tensors
