import einops
import heapq
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
import torch
from .discriminator import BaseDiscriminatorCallback


class StepBuffer:
    def __init__(self, num_envs: int, num_steps: int):
        self._env_buffer = [[] for _ in range(num_envs)]
        self.episodes = []
        self.max_steps = num_steps
        self.n_steps = 0
        self.n_updates = 0
        self._n_update_attempts = 0


    def add(self, value: torch.Tensor, done: np.ndarray):
        for env_idx, env_value in enumerate(value):
            self._env_buffer[env_idx].append(env_value)

        for env_idx in np.argwhere(done).reshape(-1):
            episode = torch.stack(self._env_buffer[env_idx])
            ep_return = episode[:, -1].sum().item()

            heapq.heappush(self.episodes, (ep_return, -self._n_update_attempts, episode))  # Number of update attempts as tiebreaker, prioritize older samples
            self._env_buffer[env_idx].clear()
            self.n_steps += len(episode)
            self._n_update_attempts += 1

            if episode is not self.episodes[0][2]:
                self.n_updates += 1

            self._trim_episodes()


    def _trim_episodes(self):
        while self.n_steps - (least_ep_len := len(self.episodes[0][2])) >= self.max_steps:
            heapq.heappop(self.episodes)
            self.n_steps -= least_ep_len


class DIRECTCallback(BaseDiscriminatorCallback):
    def __init__(self, si_buffer_size_steps: int = 8192, **kwargs):
        super().__init__(**kwargs)
        self.si_buffer_size_steps = si_buffer_size_steps


    def _init_callback(self):
        super()._init_callback()
        self.si_buffer = StepBuffer(self.training_env.num_envs, self.si_buffer_size_steps)


    def _get_rollout_steps(self) -> torch.Tensor:
        buffer: RolloutBuffer = self.model.rollout_buffer
        steps = torch.cat([
            torch.as_tensor(buffer.observations),
            torch.as_tensor(buffer.actions),
            torch.as_tensor(buffer.rewards).unsqueeze(-1)
        ], dim=-1)
        flat_steps = einops.rearrange(steps, 'buffer env dim -> (buffer env) dim')
        return flat_steps


    def _get_positive_samples(self):
        best_steps = torch.cat([ep for _, _, ep in self.si_buffer.episodes])
        return best_steps[:self.si_buffer_size_steps]


    def _get_negative_samples(self, batch_size):
        rollout_step_indices = torch.randint(0, self.model.rollout_buffer.pos, (batch_size,))
        return self._get_rollout_steps()[rollout_step_indices]


    def _on_step(self):
        obs, act, gt_reward = self._get_current_step()
        self.si_buffer.add(torch.cat([obs, act, gt_reward], dim=-1), self.locals['dones'])
        return super()._on_step()


    def _on_rollout_end(self):
        self.logger.record('direct/buffer/n_updates', self.si_buffer.n_updates)
        self.logger.record('direct/buffer/ep_rew_mean', np.mean([ret for ret, _, _ in self.si_buffer.episodes]))
        self._train_discriminator()
