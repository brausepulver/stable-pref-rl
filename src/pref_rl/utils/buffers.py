from collections import deque, defaultdict
from typing import Any

import numpy as np
import torch

from .data import SegmentDataset

EpisodeData = tuple[list[torch.Tensor], dict[str, list[Any]]]


class EpisodeBuffer:
    def __init__(self, n_envs, n_episodes, keep_all_eps: bool = False, store_step_timesteps: bool = False):
        self.n_episodes = n_episodes
        self.keep_all_eps = keep_all_eps
        self.store_step_timesteps = store_step_timesteps
        self.done_eps: deque[EpisodeData] = deque(maxlen=self.n_episodes if not self.keep_all_eps else None)
        self.running_eps: list[EpisodeData] = [([], defaultdict(list)) for _ in range(n_envs)]


    def add(self, value: torch.Tensor, done: np.ndarray, meta: dict[str, Any] = {}, timesteps: int | None = None):
        for env_idx, env_value in enumerate(value):
            ep_steps, ep_meta = self.running_eps[env_idx]
            ep_steps.append(env_value)

            if self.store_step_timesteps:
                ep_meta['timesteps'].append(timesteps)
            if 'ep_start_timesteps' not in ep_meta:
                ep_meta['ep_start_timesteps'] = timesteps
            
            for key, maybe_list in meta.items():
                try:
                    meta_value = maybe_list[env_idx]
                except TypeError:
                    meta_value = maybe_list
                ep_meta[key].append(meta_value)

        for env_idx in np.argwhere(done).reshape(-1):
            ep_steps, ep_meta = self.running_eps[env_idx]
            self.done_eps.append((torch.stack(ep_steps), dict(ep_meta)))
            self.running_eps[env_idx] = ([], defaultdict(list))


    def get_episodes(self, return_all: bool = False):
        running = [torch.stack(ep[0]) for ep in self.running_eps if ep[0]]
        done = [ep[0] for ep in self.done_eps]
        if self.keep_all_eps and not return_all:
            done = done[-self.n_episodes:]
        return done + running


    def get_episode_metas(self, return_all: bool = False):
        running = [torch.stack(ep[1]) for ep in self.running_eps if ep[1]]
        done = [ep[1] for ep in self.done_eps]
        if self.keep_all_eps and not return_all:
            done = done[-self.n_episodes:]
        return done + running


class FeedbackBuffer:
    def __init__(self, buffer_size: int, segment_size: int, segment_dimension: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.segments = torch.empty((buffer_size, 2, segment_size, segment_dimension), device=device).detach()
        self.preferences = torch.empty((buffer_size,), device=device).detach()
        self.weights = torch.empty((buffer_size,), device=device).detach()
        self.segment_metas = [None] * buffer_size
        self.position = 0
        self.size = 0


    def add(self, segments: torch.Tensor, preferences: torch.Tensor, weights: torch.Tensor, segment_metas: list | None = None) -> int:
        num_items = len(segments)
        segments = segments.to(self.device)
        preferences = preferences.to(self.device)
        weights = weights.to(self.device)
        
        if num_items > 0:
            start_pos = self.position % self.buffer_size
            end_pos = (self.position + num_items) % self.buffer_size
            
            if start_pos < end_pos:
                self.segments[start_pos:end_pos] = segments.detach()
                self.preferences[start_pos:end_pos] = preferences.detach()
                self.weights[start_pos:end_pos] = weights.detach()
                if segment_metas:
                    self.segment_metas[start_pos:end_pos] = segment_metas
            else:
                first_chunk = self.buffer_size - start_pos
                self.segments[start_pos:] = segments[:first_chunk].detach()
                self.preferences[start_pos:] = preferences[:first_chunk].detach()
                self.weights[start_pos:] = weights[:first_chunk].detach()
                if segment_metas:
                    self.segment_metas[start_pos:] = segment_metas[:first_chunk]
                if end_pos > 0:
                    self.segments[:end_pos] = segments[first_chunk:].detach()
                    self.preferences[:end_pos] = preferences[first_chunk:].detach()
                    self.weights[:end_pos] = weights[first_chunk:].detach()
                    if segment_metas:
                        self.segment_metas[:end_pos] = segment_metas[first_chunk:]

            self.position += num_items
            self.size = min(self.size + num_items, self.buffer_size)

        return num_items


    def get_dataset(self):
        return SegmentDataset(self.segments[:self.size], self.preferences[:self.size], self.weights[:self.size], self.segment_metas[:self.size])


    def clear(self):
        self.position = 0
        self.size = 0


    def __len__(self):
        return self.size
