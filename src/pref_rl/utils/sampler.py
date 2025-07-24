from typing import Callable, Union, Optional
from abc import ABC, abstractmethod

import einops
import numpy as np
import torch
import torch.nn.functional as F

from ..config import ConstantSchedule

Schedule = Union[float, Callable]


class NoValidEpisodesError(ValueError):
    pass


class BaseSamplerMetric(ABC):
    @abstractmethod
    def compute(self, split_state_actions: torch.Tensor, reward_model, **schedule_kwargs) -> torch.Tensor:
        pass


    @property
    @abstractmethod
    def name(self) -> str:
        pass


class DisagreementMetric(BaseSamplerMetric):
    @property
    def name(self) -> str:
        return 'disagreement'


    def compute(self, state_action_pairs: torch.Tensor, reward_model, **schedule_kwargs) -> torch.Tensor:
        with torch.no_grad():
            device = next(reward_model.parameters()).device
            member_rewards = reward_model(state_action_pairs.to(device))
            member_returns = einops.reduce(member_rewards, 'm n p s 1 -> m n p', 'sum')
            probabilities = F.softmax(member_returns, dim=-1)
            return probabilities[..., 0].std(dim=0)


class EntropyMetric(BaseSamplerMetric):
    @property
    def name(self) -> str:
        return 'entropy'


    def compute(self, state_action_pairs: torch.Tensor, reward_model, **schedule_kwargs) -> torch.Tensor:
        with torch.no_grad():
            device = next(reward_model.parameters()).device
            member_rewards = reward_model(state_action_pairs.to(device))
            member_returns = einops.reduce(member_rewards, 'm n p s 1 -> m n p', 'sum')
            probabilities = F.softmax(member_returns, dim=-1)
            member_entropies = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
            return member_entropies.mean(dim=0)


class CompositeMetric(BaseSamplerMetric):
    """Combines multiple metrics with weights."""


    def __init__(self, metrics: dict[str, BaseSamplerMetric], 
                    weights: Optional[dict[str, Schedule]] = None, name: str = "composite"):
        self.metrics = metrics
        weights = weights or {}
        
        self.weight_schedules = {}
        for metric_name in metrics.keys():
            weight = weights.get(metric_name, 1.0)
            self.weight_schedules[metric_name] = weight if callable(weight) else ConstantSchedule(weight)
        
        self._name = name
    

    @property
    def name(self) -> str:
        return self._name
    

    def compute(self, split_state_actions: torch.Tensor, reward_model, **schedule_kwargs) -> torch.Tensor:
        weighted_values = []
        for metric_name, metric in self.metrics.items():
            weight = self.weight_schedules[metric_name](**schedule_kwargs)
            weighted_values.append(weight * metric.compute(split_state_actions, reward_model, **schedule_kwargs))
        return torch.stack(weighted_values).sum(dim=0)


class Sampler:
    def __init__(self, segment_size: int, observation_size: int, action_size: int, 
                    sampling_metric: Optional[BaseSamplerMetric] = None, pre_sample_multiplier: int = 10):
        self.segment_size = segment_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.pre_sample_multiplier = pre_sample_multiplier
        self.sampling_metric = sampling_metric
        
        self.logging_metrics = {
            'disagreement': DisagreementMetric(),
            'entropy': EntropyMetric(),
        }


    def _get_episode_indices(self, valid_episodes: list, num_samples: int, stratified: bool = False):
        if stratified:
            episodes_count = len(valid_episodes)
            total_samples = 2 * num_samples

            repeats = torch.ones(episodes_count) * (total_samples // episodes_count)
            repeats[:total_samples % episodes_count] += 1

            ep_indices = torch.repeat_interleave(torch.arange(episodes_count), repeats.long())
            ep_indices = ep_indices[torch.randperm(len(ep_indices))]
        else:
            ep_indices = torch.randint(0, len(valid_episodes), (2 * num_samples,))
        
        return ep_indices


    def _get_segments(self, episodes: list, ep_indices: torch.Tensor):
        segments = []

        for ep_idx in ep_indices:
            ep = episodes[ep_idx]

            start_step = 0 if len(ep) == self.segment_size else np.random.randint(0, len(ep) - self.segment_size)
            offsets = torch.arange(0, self.segment_size)
            step_indices = start_step + offsets

            segment = ep[step_indices]
            segments.append(segment)

        return torch.stack(segments)


    def _sample_random_segments(self, episodes: list, num_samples: int, stratified: bool = False):
        valid_episodes = [ep for ep in episodes if len(ep) >= self.segment_size]
        if len(valid_episodes) == 0:
            raise NoValidEpisodesError('No valid episodes to sample from')

        ep_indices = self._get_episode_indices(valid_episodes, num_samples, stratified)
        segments = self._get_segments(valid_episodes, ep_indices)
        
        obs, act, gt_rewards = torch.split(segments, (self.observation_size, self.action_size, 1), dim=-1)
        state_actions = torch.cat([obs, act], dim=-1)

        state_action_pairs = einops.rearrange(state_actions, '(n p) s d -> n p s d', p=2)
        reward_pairs = einops.rearrange(gt_rewards, '(n p) s 1 -> p n s', p=2)

        return ep_indices, state_action_pairs, reward_pairs


    def compute_logging_metrics(self, state_action_pairs, reward_model, **schedule_kwargs):
        metrics = {
            metric_name: metric.compute(state_action_pairs, reward_model, **schedule_kwargs)
            for metric_name, metric in self.logging_metrics.items()
        }

        if not self.sampling_metric:
            return metrics
        
        if isinstance(self.sampling_metric, CompositeMetric):
            metrics[self.sampling_metric.name] = self.sampling_metric.compute(state_action_pairs, reward_model, **schedule_kwargs)
            for metric_name, metric in self.sampling_metric.metrics.items():
                weight = self.sampling_metric.weight_schedules[metric_name](**schedule_kwargs)
                metrics[f"{self.sampling_metric.name}_{metric_name}"] = weight * metric.compute(state_action_pairs, reward_model, **schedule_kwargs)
        elif self.sampling_metric.name not in metrics:
            metrics[self.sampling_metric.name] = self.sampling_metric.compute(state_action_pairs, reward_model, **schedule_kwargs)
        
        return metrics


    def sample_segments(self, episodes: list, episode_ages: list, num_samples: int, stratified: bool = False,
                        reward_model: Optional[Callable] = None, compute_uniform_metrics: bool = True, **schedule_kwargs):
        method = getattr(self.sampling_metric, 'name', 'uniform')

        num_samples_expanded = num_samples if method == 'uniform' else self.pre_sample_multiplier * num_samples
        ep_indices, state_action_pairs, reward_pairs = self._sample_random_segments(episodes, num_samples_expanded, stratified)

        if method != 'uniform' or compute_uniform_metrics:
            metrics = self.compute_logging_metrics(state_action_pairs, reward_model, **schedule_kwargs)
        else:
            metrics = {}

        if method == 'uniform':
            return state_action_pairs, reward_pairs, metrics

        assert self.sampling_metric
        sampling_scores = self.sampling_metric.compute(state_action_pairs, reward_model, **schedule_kwargs)
        idx = torch.topk(sampling_scores, num_samples).indices.to('cpu')
        return state_action_pairs[idx], reward_pairs[:, idx], metrics
