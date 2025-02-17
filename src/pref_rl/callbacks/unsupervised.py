from collections import deque
import einops
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
import torch
from .reward_mod import RewardModifierCallback


class UnsupervisedCallback(RewardModifierCallback):
    def __init__(self, *args, n_steps_unsuper: int = 32_000, n_epochs_unsuper: int = 50, n_neighbors: int = 5, **kwargs):
        super().__init__(*args, ep_info_key='intr_r', log_prefix='pretrain/', log_suffix='ep_intr_rew_mean', **kwargs)
        self.n_steps_unsuper = n_steps_unsuper
        self.n_epochs_unsuper = n_epochs_unsuper
        self.n_neighbors = n_neighbors


    def _init_callback(self):
        self._buffer = deque(maxlen=self.n_steps_unsuper)


    def _estimate_state_entropy(self, obs: torch.Tensor):
        all_obs = einops.rearrange(torch.stack(list(self._buffer)), 'n e d -> (n e) d')

        differences = obs.unsqueeze(1) - all_obs
        distances = torch.norm(differences, dim=-1)
        neighbor_dist = torch.kthvalue(distances, self.n_neighbors + 1, dim=-1).values

        std = np.std(self._buffer)
        normalized_dist = neighbor_dist / (std + 1e-8)
        return normalized_dist


    def _get_unnormalized_obs(self):
        if isinstance(self.training_env, VecNormalize):
            original_obs = self.model.get_vec_normalize_env().get_original_obs()
            return torch.tensor(original_obs, dtype=torch.float)

        return self._get_current_step()[0]


    def _predict_rewards(self):
        obs = self._get_unnormalized_obs()
        return self._estimate_state_entropy(obs)


    def _on_step(self):
        if self.n_steps_unsuper is None or self.num_timesteps > self.n_steps_unsuper:
            return True

        obs = self._get_unnormalized_obs()
        self._buffer.append(obs)
        return super()._on_step()


    def _run_cleanup(self):
        self._buffer = None


    def _on_rollout_end(self):
        if self.n_steps_unsuper is None or self.num_timesteps > self.n_steps_unsuper:
            self._run_cleanup()
            return

        super()._on_rollout_end()
