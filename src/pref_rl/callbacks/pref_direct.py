from collections import deque
import einops
import torch
from .discriminator import BaseDiscriminatorCallback
from .pref import BasePrefCallback


class PrefDIRECTCallback(BasePrefCallback, BaseDiscriminatorCallback):
    def __init__(self, pref_kwargs={}, direct_kwargs={}):
        super().__init__(log_prefix='discriminator/', **pref_kwargs, use_rewards_as_features=False, **direct_kwargs)
        self.rollout_steps = deque(maxlen=pref_kwargs['max_feed'])


    def _init_callback(self):
        BasePrefCallback._init_callback(self)
        BaseDiscriminatorCallback._init_callback(self)


    def _get_predictor(self):
        return self.discriminator


    def _get_steps_from_preferences(self, preferences: torch.Tensor):
        indices = (torch.arange(len(self.segment_buffer)), preferences)
        segments = self.segment_buffer[indices]
        steps = einops.rearrange(segments, 'l s d -> (l s) d')
        return steps


    def _get_positive_samples(self):
        return self._get_steps_from_preferences(self.preference_buffer)


    def _get_negative_samples(self, batch_size):
        return self._get_steps_from_preferences(1 - self.preference_buffer)


    def _train_predictor(self):
        return BaseDiscriminatorCallback._train_discriminator(self)
