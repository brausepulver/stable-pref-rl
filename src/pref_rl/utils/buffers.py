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


class FeedbackBuffer:
    def __init__(self, buffer_size: int, segment_size: int, segment_dimension: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.segments = torch.empty((buffer_size, 2, segment_size, segment_dimension), device=device).detach()
        self.preferences = torch.empty((buffer_size,), device=device).detach()
        self.weights = torch.empty((buffer_size,), device=device).detach()
        self.position = 0
        self.size = 0


    def add(self, segments: torch.Tensor, preferences: torch.Tensor, weights: torch.Tensor):
        """Add segments and preferences to the buffer with wraparound."""
        num_items = len(segments)
        segments = segments.to(self.device)
        preferences = preferences.to(self.device)
        weights = weights.to(self.device)
        
        if num_items > 0:
            start_pos = self.position % self.buffer_size
            end_pos = (self.position + num_items) % self.buffer_size
            
            if start_pos < end_pos:
                # No wraparound
                self.segments[start_pos:end_pos] = segments.detach()
                self.preferences[start_pos:end_pos] = preferences.detach()
                self.weights[start_pos:end_pos] = weights.detach()
            else:
                # Wraparound
                first_chunk = self.buffer_size - start_pos
                self.segments[start_pos:] = segments[:first_chunk].detach()
                self.preferences[start_pos:] = preferences[:first_chunk].detach()
                self.weights[start_pos:] = weights[:first_chunk].detach()
                if end_pos > 0:
                    self.segments[:end_pos] = segments[first_chunk:].detach()
                    self.preferences[:end_pos] = preferences[first_chunk:].detach()
                    self.weights[:end_pos] = weights[first_chunk:].detach()
            
            self.position += num_items
            self.size = min(self.size + num_items, self.buffer_size)
        return num_items


    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0


    def __len__(self):
        return self.size
