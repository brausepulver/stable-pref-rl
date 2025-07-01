import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import GymEnv

from ..utils.pref import EpisodeBuffer


class _EpisodeRecorderCallback(BaseCallback):
    def __init__(self, buffer: EpisodeBuffer):
        super().__init__()
        self.buffer = buffer


    def _on_step(self) -> bool:
        maybe_flat_obs = np.array(list(self.model._last_obs.values())) if isinstance(self.model._last_obs, dict) else self.model._last_obs
        obs = torch.tensor(maybe_flat_obs, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        env_actions = self.locals.get('clipped_actions', self.locals['actions'])  # These are the final actions in the same shape as passed to the environment
        act = torch.tensor(env_actions, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        gt_rewards = torch.tensor(self.locals['rewards'], dtype=torch.float).reshape(self.training_env.num_envs, -1)

        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        return True


class LoggingPPO(PPO):
    def __init__(
        self,
        policy: str | torch.nn.Module,
        env: GymEnv | str,
        *,
        save_episode_data: bool = False,
        ann_buffer_size_eps: int | None = None,
        run_id: str | None = None,
        **kwargs: any,
    ) -> None:
        super().__init__(policy, env, **kwargs)
        self.save_episode_data = save_episode_data
        self.run_id = run_id
        self.episode_buffer = EpisodeBuffer(self.n_envs, ann_buffer_size_eps)
        self._recorder = _EpisodeRecorderCallback(self.episode_buffer)


    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | None = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        if callback is None:
            callback = self._recorder
        else:
            if isinstance(callback, list):
                callback = CallbackList(callback + [self._recorder])
            else:
                callback = CallbackList([callback, self._recorder])
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_recorder", *(["episode_buffer"] if not self.save_episode_data else [])]
