from collections import deque
import einops
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch


class UnsuperCallback(BaseCallback):
    def __init__(self, *args, n_steps_unsuper=32_000, n_epochs_unsuper=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps_unsuper = n_steps_unsuper
        self.n_epochs_unsuper = n_epochs_unsuper

        self.n_neighbors = 5


    def _init_callback(self):
        self.n_epochs_model = getattr(self.model, 'n_epochs', None)
        self.model.n_epochs = self.n_epochs_unsuper

        self._buffer = deque(maxlen=self.n_steps_unsuper)


    def _on_rollout_start(self):
        self.intr_reward_buffer = []


    def _estimate_state_entropy(self, obs: torch.Tensor):
        all_obs = einops.rearrange(torch.stack(list(self._buffer)), 'n e d -> (n e) d')
        differences = obs.unsqueeze(1) - all_obs

        distances = torch.norm(differences, dim=-1)
        neighbor_dist = torch.kthvalue(distances, self.n_neighbors + 1, dim=-1).values

        return neighbor_dist.log()


    def _on_step(self):
        if self.num_timesteps > self.n_steps_unsuper:
            self.model.n_epochs = self.n_epochs_model
            return True

        obs = torch.tensor(self.model._last_obs, dtype=torch.float32)
        self._buffer.append(obs)

        state_entropy = self._estimate_state_entropy(obs)

        for env_idx in range(len(self.locals['rewards'])):
            self.locals['rewards'][env_idx] = state_entropy[env_idx]

        self.intr_reward_buffer.extend(state_entropy)

        return True


    def _on_rollout_end(self):
        if self.intr_reward_buffer:
            self.logger.record('pretrain/intr_rew_mean', np.mean(self.intr_reward_buffer))
