from abc import ABC, abstractmethod
from typing import Callable

import torch

from .sampler import Sampler
from .schedules import ConstantSchedule, BaseScheduleState, ExponentialSchedule


class BaseSynthesizer(ABC):
    """
    Abstract base for synthesizers that generate synthetic preference pairs.

    All synthesizers must implement:
      - generate_pairs: produce (segments, preferences, metrics, loss_weights)
    """


    @abstractmethod
    def generate_pairs(
        self,
        episodes: list,
        episode_ages: torch.Tensor,
        num_samples: int,
        state: BaseScheduleState,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
          segments: (N, 2, S, D)
          preferences: (N,)  float tensor with entries in {0., 1.}
          metrics: dict of torch.Tensors for logging
          loss_weights: (N,)  per-example weights
        """
        raise NotImplementedError


class TemporalSynthesizer(BaseSynthesizer):
    """
    Samples negative segments from older episodes and positive segments from newer episodes,
    based on step thresholds that can be scheduled.

    negative: episodes with age < (num_timesteps - neg_eps_until_steps)
    positive: episodes with age >= (num_timesteps - pos_eps_after_eq_steps)
    """


    def __init__(
        self,
        segment_size: int,
        observation_size: int,
        action_size: int,
        neg_eps_until_steps: int | Callable,
        pos_eps_after_eq_steps: int | Callable | None = None,
        loss_weight: float | Callable = 0.5,
    ):
        self.segment_size = segment_size
        self.observation_size = observation_size
        self.action_size = action_size

        self.neg_eps_until_steps = (
            neg_eps_until_steps if callable(neg_eps_until_steps) else ConstantSchedule(neg_eps_until_steps)
        )
        self.loss_weight = loss_weight if callable(loss_weight) else ConstantSchedule(loss_weight)

        if pos_eps_after_eq_steps is not None:
            is_schedule = callable(pos_eps_after_eq_steps)
            self.pos_eps_after_eq_steps = (
                pos_eps_after_eq_steps if is_schedule else ConstantSchedule(pos_eps_after_eq_steps)
            )
        else:
            self.pos_eps_after_eq_steps = self.neg_eps_until_steps

        self.sampler = Sampler(segment_size, observation_size, action_size)


    def _calculate_loss_weights(self, num_samples: int, state: BaseScheduleState) -> torch.Tensor:
        return torch.full((num_samples,), self.loss_weight(state.progress_remaining, state))


    def _calculate_metrics(self, prev_ages: torch.Tensor, cur_ages: torch.Tensor) -> dict:
        return {
            "synth_age_negative": prev_ages.to(torch.float),
            "synth_age_positive": cur_ages.to(torch.float),
        }


    def generate_pairs(self, episodes: list, episode_metas: list, num_samples: int, state: BaseScheduleState):
        if not episodes:
            raise ValueError("Episodes must not be empty to generate synthetic pairs")

        episode_ages = torch.tensor([ep_meta['ep_start_timesteps'] for ep_meta in episode_metas])

        prev_mask = episode_ages < (state.num_timesteps - self.neg_eps_until_steps(state.progress_remaining, state))
        cur_mask = episode_ages >= (state.num_timesteps - self.pos_eps_after_eq_steps(state.progress_remaining, state))

        prev_eps = [ep for m, ep in zip(prev_mask, episodes) if m]
        cur_eps = [ep for m, ep in zip(cur_mask, episodes) if m]

        if not prev_eps or not cur_eps:
            raise ValueError("No current or previous episodes to generate synthetic pairs from")

        prev_ep_indices, prev_segs, _, __ = self.sampler.sample_segments(prev_eps, num_samples)
        cur_ep_indices, cur_segs, _, __ = self.sampler.sample_segments(cur_eps, num_samples)

        segments = torch.stack([prev_segs, cur_segs], dim=1)
        num_segments = len(segments)
        preferences = torch.ones(num_segments, dtype=torch.float32, device=segments.device)

        loss_weights = self._calculate_loss_weights(num_samples, state)

        prev_ages = episode_ages[prev_mask][prev_ep_indices]
        cur_ages = episode_ages[cur_mask][cur_ep_indices]
        metrics = self._calculate_metrics(prev_ages, cur_ages)

        return segments, preferences, metrics, loss_weights


class FeedbackResamplingSynthesizer(BaseSynthesizer):
    """
    Resample synthetic comparisons from existing labeled pairs in FeedbackBuffer using episode-age-based
    exponential distributions that are independent of the current step and tie-aware:

    - Let groups be distinct episode ages (equal ages share the same group/rank).
    - Ranks go from 0 (oldest) to K-1 (newest).

    For loser sampling (negative side):
      weight(rank=r) ∝ exp(alpha * r), but the newest group (r = K-1) has weight 0.

    For winner sampling (positive side):
      weight(rank=r) ∝ exp(alpha * r) - exp(alpha * (K-1))  clipped at 0,
      so the oldest group has 0 and the newest group has the highest weight.
      This is equivalent to (1 - normalized_loser_weight) in spirit, but constructed to guarantee
      0 at the oldest and max at the newest.

    We assume at least 2 distinct episode groups.
    """


    def __init__(
        self,
        weight_schedule: Callable | None = None,  # for loss weights, not sampling weights
        alpha: float = 1.0,  # exponential rate over ranks
        filter_bad_orders: bool = True,
    ):
        self.weight_schedule = (
            weight_schedule if callable(weight_schedule) else ExponentialSchedule(start=1.0, end=1.0, decay=1.0)
        )
        self.alpha = float(alpha)
        self.filter_bad_orders = filter_bad_orders


    @staticmethod
    def _extract_winners_losers(segments: torch.Tensor, preferences: torch.Tensor):
        # segments: (N, 2, S, D), preferences: (N,)
        winner_idx = preferences.long()
        loser_idx = 1 - winner_idx

        batch_indices = torch.arange(len(preferences), device=segments.device)
        winners = segments[batch_indices, winner_idx]  # (N, S, D)
        losers = segments[batch_indices, loser_idx]    # (N, S, D)
        return winners, losers


    @staticmethod
    def _extract_winner_loser_ages(ages: torch.Tensor, preferences: torch.Tensor):
        # ages: (N, 2), preferences: (N,)
        winner_idx = preferences.long()
        loser_idx = 1 - winner_idx
        batch_indices = torch.arange(len(preferences), device=ages.device)
        winner_ages = ages[batch_indices, winner_idx]  # (N,)
        loser_ages = ages[batch_indices, loser_idx]    # (N,)
        return winner_ages, loser_ages


    @staticmethod
    def _episode_age_group_ranks(ep_ages: torch.Tensor):
        """
        Assign the same rank to equal episode ages (tie-aware).
        Returns group ranks in [0, K-1], where 0=oldest group and K-1=newest group.
        """
        unique_vals, inverse = torch.unique(ep_ages, sorted=True, return_inverse=True)
        # inverse ∈ [0..K-1], increasing with age
        return inverse, unique_vals.numel()


    def _exp_over_ranks(self, ranks: torch.Tensor):
        return torch.exp(self.alpha * ranks.to(torch.float32))


    def _loser_weights_from_ranks(self, ranks: torch.Tensor, K: int):
        """
        w_neg[r] ∝ exp(alpha * r), except newest group (r = K-1) has weight 0.
        """
        base = self._exp_over_ranks(-ranks)
        newest_mask = (ranks == (K - 1))
        base = torch.where(newest_mask, torch.zeros_like(base), base)
        s = base.sum()
        if s <= 0:
            raise ValueError("Unable to normalize loser weights (no eligible groups)")
        return base / s


    def _winner_weights_from_ranks(self, ranks: torch.Tensor, K: int):
        """
        w_neg[r] ∝ exp(alpha * r), except newest group (r = K-1) has weight 0.
        """
        base = self._exp_over_ranks(ranks)
        newest_mask = (ranks == 0)
        base = torch.where(newest_mask, torch.zeros_like(base), base)
        s = base.sum()
        if s <= 0:
            raise ValueError("Unable to normalize loser weights (no eligible groups)")
        return base / s


    def _calculate_loss_weights(self, num_samples: int, state: BaseScheduleState) -> torch.Tensor:
        val = self.weight_schedule(state.progress_remaining, state)
        if isinstance(val, (int, float)):
            return torch.full((num_samples,), float(val))
        return torch.ones(num_samples)


    def _calculate_metrics(self, neg_ages: torch.Tensor, pos_ages: torch.Tensor) -> dict:
        return {
            "synth_age_negative": neg_ages.to(torch.float32),
            "synth_age_positive": pos_ages.to(torch.float32),
        }


    def generate_pairs(self, episodes: list, episode_metas: list, num_samples: int, state: BaseScheduleState):
        fb = state.feed_buffer
        if fb is None or len(fb) == 0:
            raise ValueError("FeedbackResamplingSynthesizer requires a non-empty FeedbackBuffer")

        # Use only the filled portion
        segments = fb.segments[:fb.size]        # (M, 2, S, D)
        preferences = fb.preferences[:fb.size]  # (M,)
        seg_ages = torch.tensor([[meta['ep_start_timesteps'] for meta in meta_pair] for meta_pair in fb.segment_metas[:fb.size]])    # (M, 2)

        winners, losers = self._extract_winners_losers(segments, preferences)
        winner_ages, loser_ages = self._extract_winner_loser_ages(seg_ages, preferences)

        # We assume at least 2 distinct episode groups
        # Build tie-aware ranks on loser ages (for negatives) and on winner ages (for positives)
        loser_ranks, K_neg = self._episode_age_group_ranks(loser_ages)
        winner_ranks, K_pos = self._episode_age_group_ranks(winner_ages)

        if K_neg < 2 or K_pos < 2:
            raise ValueError("Need at least 2 distinct episode groups for resampling")

        # Step-independent exponential distributions over ranks
        neg_weights = self._loser_weights_from_ranks(loser_ranks, K_neg)
        pos_weights = self._winner_weights_from_ranks(winner_ranks, K_pos)

        # Sample indices with replacement according to weights
        num_samples_expanded = 10 * num_samples if self.filter_bad_orders else num_samples
        neg_indices = torch.multinomial(neg_weights, num_samples=num_samples_expanded, replacement=True)
        pos_indices = torch.multinomial(pos_weights, num_samples=num_samples_expanded, replacement=True)

        sampled_neg = losers[neg_indices]           # (N, S, D)
        sampled_pos = winners[pos_indices]          # (N, S, D)
        sampled_neg_ages = loser_ages[neg_indices]  # (N,)
        sampled_pos_ages = winner_ages[pos_indices] # (N,)

        pairs = torch.stack([sampled_neg, sampled_pos], dim=1)  # (N, 2, S, D)

        if self.filter_bad_orders:
            keep_pair_indices = sampled_neg_ages < sampled_pos_ages
            selected_pairs = pairs[keep_pair_indices][:num_samples]
        else:
            selected_pairs = pairs

        # Build pair tensor: [negative, positive] and labels
        preferences_out = torch.ones(len(selected_pairs), dtype=torch.float32, device=pairs.device)

        loss_weights = self._calculate_loss_weights(len(selected_pairs), state)
        metrics = self._calculate_metrics(sampled_neg_ages, sampled_pos_ages)

        return selected_pairs, preferences_out, metrics, loss_weights
