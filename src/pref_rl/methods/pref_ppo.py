import einops
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RewardEnsemble(nn.Module):
    def __init__(self, input_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([self._build_member(input_dim, **kwargs) for _ in range(ensemble_size)])


    def _build_member(self, input_dim, net_arch=[256, 256, 256], activation_fn=nn.LeakyReLU, output_fn=nn.Tanh):
        return nn.Sequential(*(
            [nn.Linear(input_dim, net_arch[0])] +
            sum([[activation_fn, nn.Linear(_from, _to)] for _from, _to in zip(net_arch, net_arch[1:] + [1])], []) +
            [output_fn]
        ))


    def forward(self, x):
        return torch.cat([member(x) for member in self.members], dim=-1).mean(dim=-1)


class PrefCallback(BaseCallback):
    def __init__(self, feed_type=1, n_steps_reward=32_000, ep_buffer_size=100, segment_size=50, max_feed=2_000, feed_batch_size=200, n_epochs_reward=10, learning_rate_reward=3e-4, batch_size_reward=128, reward_model_kwargs={}, teacher={'beta': -1, 'gamma': 1, 'eps_mistake': 0, 'eps_skip': 0, 'eps_equal': 0}, **kwargs):
        super().__init__(**kwargs)

        self.n_steps_reward = n_steps_reward
        self.ep_buffer_size = ep_buffer_size
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.n_epochs_reward = n_epochs_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward


    def _init_callback(self):
        input_dim = self.training_env.observation_space.shape[0] + self.training_env.action_space.shape[0]

        self.reward_model = RewardEnsemble(input_dim, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.segment_buffer = torch.empty((0, 2, self.segment_size, input_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)

        self.num_feed = 0


    def _on_training_start(self):
        ep_steps = self.training_env.envs[0].spec.max_episode_steps * self.ep_buffer_size
        buffer_size = int(ep_steps / self.training_env.num_envs)  # ReplayBuffer stores num_envs episodes in parallel

        self.annotation_buffer = ReplayBuffer(
            buffer_size,
            self.training_env.observation_space,
            self.training_env.action_space,
            n_envs=self.training_env.num_envs
        )


    def _on_rollout_start(self):
        self.pred_reward_buffer = [[] for _ in range(self.training_env.num_envs)]


    def _sample_segments(self, num_samples: int):
        buffer_fill = self.annotation_buffer.pos if not self.annotation_buffer.full else self.annotation_buffer.buffer_size
        start_indices = torch.randint(0, buffer_fill - self.segment_size, (num_samples,))

        offsets = torch.arange(0, self.segment_size).expand((num_samples, -1))
        step_indices = einops.repeat(start_indices, 'n -> n s', s=self.segment_size) + offsets

        env_indices = torch.randint(0, self.annotation_buffer.n_envs, (num_samples,))
        env_indices = einops.repeat(env_indices, 'n -> n s', s=self.segment_size)

        obs = torch.tensor(self.annotation_buffer.observations[step_indices, env_indices], dtype=torch.float32)
        act = torch.tensor(self.annotation_buffer.actions[step_indices, env_indices], dtype=torch.float32)
        gt_rewards = torch.tensor(self.annotation_buffer.rewards[step_indices, env_indices], dtype=torch.float32)

        return torch.cat([obs, act], dim=-1), gt_rewards


    def _query_segments(self, left: torch.Tensor, right: torch.Tensor, left_rew: torch.Tensor, right_rew: torch.Tensor):
        return left_rew.sum(dim=-1) < right_rew.sum(dim=-1)


    def _expand_data(self):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        segments, gt_rewards = self._sample_segments(num_samples)

        left, right = einops.rearrange(segments, '(n1 n2) s d -> n1 n2 s d', n1=2)
        left_rew, right_rew = einops.rearrange(gt_rewards, '(n1 n2) s -> n1 n2 s', n1=2)

        preferences = self._query_segments(left, right, left_rew, right_rew)

        self.segment_buffer = torch.cat([self.segment_buffer, torch.stack([left, right], dim=1)])
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])

        self.num_feed += num_samples
        self.logger.record('reward_model/num_feed', self.num_feed)


    def _calc_accuracy(self, pred_returns, preferences):
        pred_preferences = torch.argmax(pred_returns, dim=1)

        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _train_reward_model(self):
        self.reward_model.train()

        dataset = TensorDataset(self.segment_buffer, self.preference_buffer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True)

        batch_losses = []
        batch_accuracies = []

        for _ in range(self.n_epochs_reward):
            for segments, preferences in dataloader:
                self.rew_optimizer.zero_grad()

                pred_rewards = self.reward_model(segments)
                pred_returns = einops.reduce(pred_rewards, 'b n s -> b n', 'sum')

                loss = self.rew_loss(pred_returns, preferences)
                accuracy = self._calc_accuracy(pred_returns, preferences)

                loss.backward()
                self.rew_optimizer.step()

                batch_losses.append(loss.item())
                batch_accuracies.append(accuracy)

        self.logger.record('reward_model/loss', np.mean(batch_losses))
        self.logger.record('reward_model/accuracy', np.mean(batch_accuracies))


    def _store_pred_rewards(self, pred_rewards, rewards):
        for env_idx in range(len(pred_rewards)):
            pred_reward = pred_rewards[env_idx].item()

            rewards[env_idx] = pred_reward  # We cannot modify the reference in self.locals['rewards'] directly
            self.pred_reward_buffer[env_idx] = pred_reward


    def _update_ep_info(self, infos, ep_pred_rewards):
        for env_idx, info in enumerate(infos):
            if ep_info := info.get('episode'):
                ep_mean_pred_r = np.mean(ep_pred_rewards[env_idx])
                ep_info['pred_r'] = ep_mean_pred_r
                ep_pred_rewards[env_idx] = []


    def _on_step(self):
        self.reward_model.eval()

        if self.num_timesteps % self.n_steps_reward == 0 and self.num_feed < self.max_feed:
            self._expand_data()
            self._train_reward_model()

        obs = self.model._last_obs
        act = self.locals['actions']
        gt_rewards = self.locals['rewards']

        self.annotation_buffer.add(obs, self.locals['new_obs'], act, gt_rewards, self.locals['dones'], self.locals['infos'])

        state_actions = torch.cat([torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)], dim=-1)

        with torch.no_grad():
            pred_rewards = self.reward_model(state_actions)
            self._store_pred_rewards(pred_rewards, self.locals['rewards'])

        self._update_ep_info(self.locals['infos'], self.pred_reward_buffer)

        return True


    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) == 0:
            return

        ep_pred_rew_mean = np.mean([ep_info['pred_r'] for ep_info in self.model.ep_info_buffer])
        self.logger.record('reward_model/ep_pred_rew_mean', ep_pred_rew_mean)


class PrefPPO(PPO):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.pref_callback = PrefCallback(**pref)


    def learn(self, *args, callback=None, **kwargs):
        callback = ([callback] if callback is not None else []) + [self.pref_callback]
        callback = CallbackList(callback)
        return super().learn(*args, callback=callback, **kwargs)


    def _excluded_save_params(self):
        return ["pref_callback"] + super()._excluded_save_params()
