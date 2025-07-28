import torch
from typing import Optional, Union, Callable

from .sampler import Sampler
from .schedules import ConstantSchedule, ScheduleState


class TemporalSynthesizer:
    def __init__(
            self,
            segment_size: int,
            observation_size: int,
            action_size: int, 
            neg_eps_until_steps: Union[int, Callable],
            pos_eps_after_eq_steps: Optional[Union[int, Callable]] = None,
            loss_weight: Union[float, Callable] = 0.5,
        ):
        self.neg_eps_until_steps = neg_eps_until_steps if callable(neg_eps_until_steps) else ConstantSchedule(neg_eps_until_steps)
        self.loss_weight = loss_weight if callable(loss_weight) else ConstantSchedule(loss_weight)

        if pos_eps_after_eq_steps is not None:
            is_schedule = callable(pos_eps_after_eq_steps)
            self.pos_eps_after_eq_steps = pos_eps_after_eq_steps if is_schedule else ConstantSchedule(pos_eps_after_eq_steps)
        else:
            self.pos_eps_after_eq_steps = self.neg_eps_until_steps

        self.sampler = Sampler(segment_size, observation_size, action_size)


    def _calculate_loss_weights(self, num_samples: int, state: ScheduleState) -> torch.Tensor:
        return torch.full((num_samples,), self.loss_weight(state.progress_remaining, state))


    def _calculate_metrics(self, prev_ages: torch.Tensor, cur_ages: torch.Tensor) -> dict:
        return {
            'synth_age_negative': prev_ages.to(torch.float),
            'synth_age_positive': cur_ages.to(torch.float),
        }


    def generate_pairs(self, episodes: list, episode_ages: torch.Tensor, num_samples: int, state: ScheduleState):
        if not episodes:
            raise ValueError("Episodes must not be empty to generate synthetic pairs")

        prev_mask = episode_ages < (state.num_timesteps - self.neg_eps_until_steps(state.progress_remaining, state))
        cur_mask = episode_ages >= (state.num_timesteps - self.pos_eps_after_eq_steps(state.progress_remaining, state))

        prev_eps = [ep for m, ep in zip(prev_mask, episodes) if m]
        cur_eps = [ep for m, ep in zip(cur_mask, episodes) if m]

        if not prev_eps or not cur_eps:
            raise ValueError("No current or previous episodes to generate synthetic pairs from")

        prev_ep_indices, prev_segs, prev_rewards = self.sampler.sample_segments(prev_eps, num_samples)
        cur_ep_indices, cur_segs, cur_rewards = self.sampler.sample_segments(cur_eps, num_samples)

        segments = torch.stack([prev_segs, cur_segs], dim=1)
        num_segments = len(segments)
        preferences = torch.ones(num_segments, dtype=torch.float32)

        loss_weights = self._calculate_loss_weights(num_samples, state)
        metrics = self._calculate_metrics(episode_ages[prev_mask], episode_ages[cur_mask])
        return segments, preferences, metrics, loss_weights
