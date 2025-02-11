import einops
import gymnasium as gym
import heapq
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override
from ..utils import build_layered_module
from .reward_mod import RewardModifierCallback


class Discriminator(nn.Module):
    def __init__(self, input_dim, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__()
        self.layers = build_layered_module(input_dim, net_arch, activation_fn)


    def forward(self, x):
        return self.layers(x)


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


class DIRECTCallback(RewardModifierCallback):
    def __init__(self,
        si_buffer_size_steps: int = 8192,
        n_epochs_disc: int = 10,
        learning_rate_disc: float = 2e-4,
        batch_size_disc: int = 8192,
        disc_kwargs: dict = {},
        reward_mixture_coef: float = 0.5,
        **kwargs
    ):
        super().__init__(log_prefix='direct/', **kwargs)

        self.si_buffer_size_steps = si_buffer_size_steps
        self.n_epochs_disc = n_epochs_disc
        self.lr_disc = learning_rate_disc
        self.batch_size_disc = batch_size_disc
        self.disc_kwargs = disc_kwargs
        self.reward_mixture_coef = reward_mixture_coef


    def _init_callback(self):
        self.si_buffer = StepBuffer(self.training_env.num_envs, self.si_buffer_size_steps)

        obs_dim = self.training_env.observation_space.shape[0]
        act_dim = 1 if isinstance(self.training_env.action_space, gym.spaces.Discrete) else self.training_env.action_space.shape[0]
        reward_dim = 1

        input_dim = obs_dim + act_dim + reward_dim
        self.discriminator = Discriminator(input_dim, **self.disc_kwargs)
        self.disc_loss = nn.BCEWithLogitsLoss()
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)


    def _get_current_step(self):
        obs = torch.as_tensor(self.model._last_obs, dtype=torch.float).reshape(self.training_env.num_envs, -1)
        act = torch.as_tensor(self.locals['actions'], dtype=torch.float).reshape(self.training_env.num_envs, -1)
        gt_reward = torch.as_tensor(self.locals['rewards'], dtype=torch.float).reshape(self.training_env.num_envs, -1)
        return obs, act, gt_reward


    def _get_rollout_steps(self) -> torch.Tensor:
        buffer: RolloutBuffer = self.model.rollout_buffer
        steps = torch.cat([torch.as_tensor(buffer.observations), torch.as_tensor(buffer.actions), torch.as_tensor(buffer.rewards).unsqueeze(-1)], dim=-1)
        flat_steps = einops.rearrange(steps, 'buffer env dim -> (buffer env) dim')
        return flat_steps


    def _build_dataset(self) -> TensorDataset:
        best_steps = torch.cat([ep for _, _, ep in self.si_buffer.episodes])[:self.si_buffer_size_steps]

        rollout_step_indices = torch.randint(0, self.model.rollout_buffer.pos, (self.si_buffer_size_steps,))
        rollout_steps = self._get_rollout_steps()[rollout_step_indices]

        steps = torch.cat([best_steps, rollout_steps])
        labels = torch.cat([torch.ones(len(best_steps)), torch.zeros(len(rollout_steps))])
        return TensorDataset(steps, labels)


    def _compute_disc_loss(self, steps: torch.Tensor, labels: torch.Tensor):
        logits = self.discriminator(steps).squeeze()
        loss = self.disc_loss(logits, labels)

        pred_labels = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (pred_labels == labels).float().mean().item()

        return loss, accuracy


    def _train_discriminator(self):
        self.discriminator.train()

        dataloader = DataLoader(self._build_dataset(), batch_size=self.batch_size_disc, shuffle=True)
        losses = []
        accuracies = []

        for _ in range(self.n_epochs_disc):
            for steps, labels in dataloader:
                self.disc_optimizer.zero_grad()

                loss, accuracy = self._compute_disc_loss(steps, labels)
                loss.backward()
                self.disc_optimizer.step()

                losses.append(loss.item())
                accuracies.append(accuracy)

        self.logger.record('direct/discriminator/train/loss', np.mean(losses))
        self.logger.record('direct/discriminator/train/accuracy', np.mean(accuracies))


    @override
    def _predict_rewards(self):
        self.discriminator.eval()

        obs, act, gt_reward = self._get_current_step()

        with torch.no_grad():
            cat_step = torch.cat([obs, act, gt_reward], dim=-1)
            disc_reward = self.reward_mixture_coef * self.discriminator(cat_step)

        return disc_reward + (1 - self.reward_mixture_coef) * gt_reward


    def _on_step(self):
        obs, act, gt_reward = self._get_current_step()
        self.si_buffer.add(torch.cat([obs, act, gt_reward], dim=-1), self.locals['dones'])
        return super()._on_step()


    def _on_rollout_end(self):
        self.logger.record('direct/buffer/n_updates', self.si_buffer.n_updates)
        self.logger.record('direct/buffer/ep_rew_mean', np.mean([ret for ret, _, _ in self.si_buffer.episodes]))
        self._train_discriminator()
