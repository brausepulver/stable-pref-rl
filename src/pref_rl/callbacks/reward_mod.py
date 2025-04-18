from abc import abstractmethod
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch


class RewardModifierCallback(BaseCallback):
    def __init__(self, ep_info_key: str = 'pred_r', log_prefix: str = 'rollout/', log_suffix = 'ep_pred_rew_mean', **kwargs):
        super().__init__(**kwargs)
        self.ep_info_key = ep_info_key
        self.log_prefix = log_prefix
        self.log_suffix = log_suffix


    def _get_input_sizes(self):
        def normalize_shape(shape):
            return (1,) if len(shape) == 0 else shape

        obs_space = self.training_env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            observation_size = sum([normalize_shape(space.shape)[0] for space in obs_space.spaces.values()])
        else:
            observation_size = normalize_shape(obs_space.shape)[0]

        action_size = normalize_shape(self.training_env.action_space.shape)[0]
        return observation_size, action_size


    def _get_current_step(self):
        maybe_flat_obs = np.array(list(self.model._last_obs.values())) if isinstance(self.model._last_obs, dict) else self.model._last_obs
        obs = torch.tensor(maybe_flat_obs, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        env_actions = self.locals.get('clipped_actions', self.locals['actions'])  # These are the final actions in the same shape as passed to the environment
        act = torch.tensor(env_actions, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        gt_rewards = torch.tensor(self.locals['rewards'], dtype=torch.float).reshape(self.training_env.num_envs, -1)
        return obs, act, gt_rewards


    def _on_rollout_start(self):
        self.pred_reward_buffer = [[] for _ in range(self.training_env.num_envs)]


    @abstractmethod
    def _predict_rewards(self) -> torch.Tensor:
        raise NotImplementedError


    def _store_rewards(self, pred_rewards: torch.Tensor):
        for env_idx in range(len(pred_rewards)):
            pred_reward = pred_rewards[env_idx].item()

            self.locals['rewards'][env_idx] = pred_reward  # We cannot modify the reference in self.locals['rewards'] directly
            self.pred_reward_buffer[env_idx].append(pred_reward)


    def _save_returns(self, infos: list):
        for env_idx, info in enumerate(infos):
            ep_info = info.get('episode')
            if ep_info is None:
                continue

            ep_pred_return = sum(self.pred_reward_buffer[env_idx])
            ep_info[self.ep_info_key] = ep_pred_return

            ep_pred_rew = np.mean(self.pred_reward_buffer[env_idx])
            ep_info[f"{self.ep_info_key}_step"] = ep_pred_rew

            if len(self.pred_reward_buffer[env_idx]) > 1:
                ep_pred_std = np.std(self.pred_reward_buffer[env_idx])
            else:
                ep_pred_std = 0.0
            ep_info[f"{self.ep_info_key}_std"] = ep_pred_std

            ep_info[f"{self.ep_info_key}_coef_var"] = ep_pred_std / ep_pred_rew
                
            self.pred_reward_buffer[env_idx] = []


    def _on_step(self):
        pred_rewards = self._predict_rewards()
        self._store_rewards(pred_rewards)
        self._save_returns(self.locals['infos'])
        return True
