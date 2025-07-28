import torch
from typing import Optional

from .sampler import Sampler


class TemporalSynthesizer:
    def __init__(
            self,
            segment_size: int,
            observation_size: int,
            action_size: int, 
            neg_eps_until_steps: int,
            pos_eps_after_eq_steps: Optional[int] = None,
            loss_weight: float = 0.5,
        ):
        self.neg_eps_until_steps = neg_eps_until_steps
        self.pos_eps_after_eq_steps = pos_eps_after_eq_steps or self.neg_eps_until_steps
        self.loss_weight = loss_weight

        self.sampler = Sampler(segment_size, observation_size, action_size)


    def _calculate_loss_weights(self, num_samples: int) -> torch.Tensor:
        return torch.full((num_samples,), self.loss_weight)


    def _calculate_metrics(self, prev_ages: torch.Tensor, cur_ages: torch.Tensor) -> dict:
        return {
            'synth_age_negative': prev_ages.to(torch.float),
            'synth_age_positive': cur_ages.to(torch.float),
        }


    def generate_pairs(self, episodes: list, episode_ages: torch.Tensor, num_samples: int, timesteps: int):
        if not episodes:
            raise ValueError("Episodes must not be empty to generate synthetic pairs")

        prev_mask = episode_ages < (timesteps - self.neg_eps_until_steps)
        cur_mask = episode_ages >= (timesteps - self.pos_eps_after_eq_steps)

        prev_eps = [ep for m, ep in zip(prev_mask, episodes) if m]
        cur_eps = [ep for m, ep in zip(cur_mask, episodes) if m]

        if not prev_eps or not cur_eps:
            raise ValueError("No current or previous episodes to generate synthetic pairs from")

        prev_ep_indices, prev_segs, prev_rewards = self.sampler.sample_segments(prev_eps, num_samples)
        cur_ep_indices, cur_segs, cur_rewards = self.sampler.sample_segments(cur_eps, num_samples)

        segments = torch.stack([prev_segs, cur_segs], dim=1)
        num_segments = len(segments)
        preferences = torch.ones(num_segments, dtype=torch.float32)

        loss_weights = self._calculate_loss_weights(num_samples)
        metrics = self._calculate_metrics(episode_ages[prev_mask], episode_ages[cur_mask])
        return segments, preferences, metrics, loss_weights
