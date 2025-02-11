from abc import abstractmethod
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
import torch


class RewardModifierCallback(EventCallback):
    def __init__(self, log_prefix='rollout/', **kwargs):
        super().__init__(**kwargs)
        self.log_prefix = log_prefix


    def _get_current_step(self):
        obs = torch.tensor(self.model._last_obs, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        act = torch.tensor(self.locals['actions'], dtype=torch.float).reshape(self.training_env.num_envs, -1)
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

            ep_mean_pred_r = np.mean(self.pred_reward_buffer[env_idx])
            ep_info['pred_r'] = ep_mean_pred_r
            self.pred_reward_buffer[env_idx] = []


    def _on_step(self):
        pred_rewards = self._predict_rewards()
        self._store_rewards(pred_rewards)
        self._save_returns(self.locals['infos'])
        return True


    def _on_rollout_end(self):
        ep_pred_rew = [ep_info['pred_r'] for ep_info in self.model.ep_info_buffer if 'pred_r' in ep_info]

        if ep_pred_rew:
            self.logger.record(f"{self.log_prefix}ep_pred_rew_mean", np.mean(ep_pred_rew))
