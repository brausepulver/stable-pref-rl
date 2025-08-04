from typing import Callable
from abc import ABC, abstractmethod

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import ConstantSchedule, BaseScheduleState


class NoValidEpisodesError(ValueError):
    pass


class BaseSamplerMetric(ABC):
    @abstractmethod
    def compute(self, state_action_pairs: torch.Tensor, reward_model: nn.Module, schedule_state: BaseScheduleState) -> dict[str, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BaseSamplerFilter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, state_action_pairs: torch.Tensor, reward_model: nn.Module, schedule_state: BaseScheduleState) -> torch.Tensor:
        pass


class DisagreementMetric(BaseSamplerMetric):
    @property
    def name(self) -> str:
        return 'disagreement'

    def compute(self, state_action_pairs, reward_model, schedule_state):
        with torch.no_grad():
            device = next(reward_model.parameters()).device
            member_rewards = reward_model(state_action_pairs.to(device))

        member_returns = einops.reduce(member_rewards, 'm n p s 1 -> m n p', 'sum')
        probabilities = F.softmax(member_returns, dim=-1)
        values = probabilities[..., 0].std(dim=0)
        return {self.name: values}


class EntropyMetric(BaseSamplerMetric):
    @property
    def name(self) -> str:
        return 'entropy'

    def compute(self, state_action_pairs, reward_model, schedule_state):
        with torch.no_grad():
            device = next(reward_model.parameters()).device
            member_rewards = reward_model(state_action_pairs.to(device))

        member_returns = einops.reduce(member_rewards, 'm n p s 1 -> m n p', 'sum')
        probabilities = F.softmax(member_returns, dim=-1)
        member_entropies = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
        values = member_entropies.mean(dim=0)
        return {self.name: values}


class CompositeMetric(BaseSamplerMetric):
    """Combines multiple metrics with weights."""

    def __init__(self, metrics: dict[str, BaseSamplerMetric], weights: dict[str, float | Callable] = {}, name: str = "composite"):
        self.metrics = metrics
        self._name = name

        self.weights = {}
        for metric_name in metrics.keys():
            weight = weights.get(metric_name, 1.0)
            self.weights[metric_name] = weight if callable(weight) else ConstantSchedule(weight)

    @property
    def name(self) -> str:
        return self._name

    def _compute_metrics(self, state_action_pairs: torch.Tensor, reward_model: nn.Module, schedule_state: BaseScheduleState):
        metrics = {}
        for metric in self.metrics.values():
            metrics |= metric.compute(state_action_pairs, reward_model, schedule_state)
        return metrics

    def _aggregate_metrics(self, computed_metrics: dict[str, torch.Tensor], schedule_state: BaseScheduleState):
        weighted_values = []
        for name, value in computed_metrics.items():
            weight = self.weights[name](schedule_state.progress_remaining, schedule_state)
            weighted_values.append(weight * value)
        weighted_value = torch.stack(weighted_values).sum(dim=0)
        return weighted_value

    def compute(self, state_action_pairs, reward_model, schedule_state):
        metrics = self._compute_metrics(state_action_pairs, reward_model, schedule_state)
        metrics[self.name] = self._aggregate_metrics(metrics, schedule_state)  # Add the composite metric itself
        return metrics


class Sampler:
    def __init__(
        self,
        segment_size: int,
        observation_size: int,
        action_size: int,
        sampling_metric: BaseSamplerMetric | None = None,
        pre_sample_multiplier: int = 10,
        logging_metrics: list[BaseSamplerMetric] | None = None,
        filters: list[BaseSamplerFilter] | None = None
    ):
        self.segment_size = segment_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.pre_sample_multiplier = pre_sample_multiplier
        self.sampling_metric = sampling_metric
        self.logging_metrics = logging_metrics or {}
        self.filters = filters or []


    def _get_episode_indices(self, valid_episodes: list, num_samples: int, stratified: bool = False):
        if stratified:
            episodes_count = len(valid_episodes)
            repeats = torch.ones(episodes_count) * (num_samples // episodes_count)
            repeats[:num_samples % episodes_count] += 1

            ep_indices = torch.repeat_interleave(torch.arange(episodes_count), repeats.long())
            ep_indices = ep_indices[torch.randperm(len(ep_indices))]
        else:
            ep_indices = torch.randint(0, len(valid_episodes), (num_samples,))

        return ep_indices


    def _get_segments(self, episodes: list, episode_metas: list, ep_indices: torch.Tensor):
        segments = []
        segment_metas = []

        for ep_idx in ep_indices:
            ep = episodes[ep_idx]
            ep_meta = episode_metas[ep_idx]

            start_step = 0 if len(ep) == self.segment_size else np.random.randint(0, len(ep) - self.segment_size)
            offsets = torch.arange(0, self.segment_size)
            step_indices = start_step + offsets

            segment = ep[step_indices]
            segments.append(segment)

            segment_meta = {}
            for key, values in ep_meta.items():
                try:
                    segment_meta[key] = values[start_step:start_step + self.segment_size]
                except TypeError:
                    segment_meta[key] = values
            segment_metas.append(segment_meta)

        return torch.stack(segments), segment_metas


    def sample_segments(self, episodes: list, episode_metas: list, num_samples: int, stratified: bool = False):
        valid_episodes = [ep for ep in episodes if len(ep) >= self.segment_size]
        valid_episode_metas = [episode_metas[i] for i, ep in enumerate(episodes) if len(ep) >= self.segment_size]
        if len(valid_episodes) == 0:
            raise NoValidEpisodesError('No valid episodes to sample from')

        ep_indices = self._get_episode_indices(valid_episodes, num_samples, stratified)
        segments, segment_metas = self._get_segments(valid_episodes, valid_episode_metas, ep_indices)

        obs, act, gt_rewards = torch.split(segments, [self.observation_size, self.action_size, 1], dim=-1)
        state_actions = torch.cat([obs, act], dim=-1)
        return state_actions, gt_rewards, segment_metas


    def compute_logging_metrics(self, metrics: dict[str, torch.Tensor], state_action_pairs: torch.Tensor, reward_model: nn.Module, schedule_state: BaseScheduleState | None = None):
        for metric in self.logging_metrics:
            if metric.name not in metrics:
                metrics |= metric.compute(state_action_pairs, reward_model, schedule_state)

        return metrics


    def _apply_filters(self, state_action_pairs: torch.Tensor, reward_model: nn.Module | None, schedule_state: BaseScheduleState | None) -> torch.Tensor:
        device = state_action_pairs.device
        keep_mask = torch.ones(state_action_pairs.shape[0], dtype=torch.bool, device=device)
        for flt in self.filters:
            mask = flt.compute(state_action_pairs, reward_model, schedule_state)
            keep_mask &= mask

        kept_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        return kept_indices


    def sample_pairs(
        self,
        episodes: list,
        episode_metas: list,
        num_samples: int,
        stratified: bool = False,
        reward_model: Callable | None = None,
        schedule_state: BaseScheduleState | None = None,
        log_metrics: bool = True
    ):
        method = getattr(self.sampling_metric, 'name', 'uniform')

        num_samples_expanded = num_samples if method == 'uniform' else self.pre_sample_multiplier * num_samples
        num_segments = 2 * num_samples_expanded
        state_actions, gt_rewards, segment_metas = self.sample_segments(episodes, episode_metas, num_segments, stratified)

        state_action_pairs = einops.rearrange(state_actions, '(n p) s d -> n p s d', p=2)
        reward_pairs = einops.rearrange(gt_rewards, '(n p) s 1 -> p n s', p=2)
        segment_meta_pairs = []
        for i in range(0, len(segment_metas), 2):
            segment_meta_pairs.append([segment_metas[i], segment_metas[i + 1]])

        keep_indices = self._apply_filters(state_action_pairs, reward_model, schedule_state)
        state_action_pairs = state_action_pairs[keep_indices]
        reward_pairs = reward_pairs[:, keep_indices]
        segment_meta_pairs = [segment_meta_pairs[i] for i in keep_indices.tolist()]

        metrics = {}
        if method != 'uniform':
            assert self.sampling_metric
            metrics = self.sampling_metric.compute(state_action_pairs, reward_model, schedule_state)

        if log_metrics:
            metrics = self.compute_logging_metrics(metrics, state_action_pairs, reward_model, schedule_state)

        if method == 'uniform' or state_action_pairs.shape[0] <= num_samples:
            return state_action_pairs, reward_pairs, metrics, segment_meta_pairs

        idx = torch.topk(metrics[self.sampling_metric.name], num_samples).indices.to('cpu')
        selected_segment_meta_pairs = [segment_meta_pairs[i] for i in idx]
        return state_action_pairs[idx], reward_pairs[:, idx], metrics, selected_segment_meta_pairs
