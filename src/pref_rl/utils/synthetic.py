import numpy as np
import torch
from typing import Optional, Literal

from .sampler import Sampler


class SyntheticTeacher:
    def __init__(
            self,
            segment_size: int,
            observation_size: int,
            action_size: int, 
            limit_prev_recent_steps,
            limit_cur_recent_steps: Optional[int] = None,
            weight_scheme: Literal['constant', 'temporal', 'pred_return'] = 'constant',
            max_weight: float = 0.5,
        ):
        self.limit_prev_recent_steps = limit_prev_recent_steps
        self.limit_cur_recent_steps = limit_cur_recent_steps or limit_prev_recent_steps
        self.weight_scheme = weight_scheme
        self.max_weight = max_weight

        self.sampler = Sampler(segment_size, observation_size, action_size)
        

    def _calculate_ep_ages(self, episodes: list, timesteps: int) -> torch.Tensor:
        episode_lengths = [len(ep) for ep in episodes]
        cumulative_lengths = np.cumsum([0] + episode_lengths)
        ages = timesteps - cumulative_lengths[:-1] - torch.tensor(episode_lengths) / 2
        return torch.from_numpy(ages)


    def _calculate_weights(self, num_samples: int) -> torch.Tensor:
        return torch.full((num_samples,), self.max_weight)


    def _calculate_metrics(self, prev_ages: torch.Tensor, cur_ages: torch.Tensor) -> dict:
        diff = (prev_ages.mean() - cur_ages.mean()).item() if prev_ages.numel() and cur_ages.numel() else 0.0
        return {'synth_avg_age_diff': diff}


    def generate_pairs(self, episodes: list, num_samples: int, timesteps: int):
        if not episodes:
            raise ValueError("Episodes must not be empty to generate synthetic pairs")

        ages = self._calculate_ep_ages(episodes, timesteps)

        prev_mask = ages > (timesteps - self.limit_prev_recent_steps)
        cur_mask = ages <= (timesteps - self.limit_cur_recent_steps)
        prev_eps = [ep for m, ep in zip(prev_mask, episodes) if m]
        cur_eps = [ep for m, ep in zip(cur_mask, episodes) if m]

        if not prev_eps or not cur_eps:
            raise ValueError("No current or previous episodes to generate synthetic pairs from")

        prev_ep_indices, prev_segs, prev_rewards = self.sampler.sample_segments(prev_eps, num_samples)
        cur_ep_indices, cur_segs, cur_rewards = self.sampler.sample_segments(cur_eps, num_samples)

        segments = torch.stack([prev_segs, cur_segs], dim=1)
        num_segments = len(segments)
        preferences = torch.zeros(num_segments, dtype=torch.float32)

        weights = self._calculate_weights(num_samples)
        metrics = self._calculate_metrics(ages[prev_mask], ages[cur_mask])
        return segments, preferences, metrics, weights
