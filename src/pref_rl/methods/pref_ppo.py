import einops
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch
import torch.nn as nn


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
    def __init__(self, feed_type=1, n_steps_reward=32_000, segment_size=50, max_feed=2_000, feed_batch_size=200, n_epochs_reward=10, learning_rate_reward=3e-4, reward_model_kwargs={}, teacher={'beta': -1, 'gamma': 1, 'eps_mistake': 0, 'eps_skip': 0, 'eps_equal': 0}, **kwargs):
        super().__init__(**kwargs)

        self.n_steps_reward = n_steps_reward
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.n_epochs_reward = n_epochs_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward


    def _init_callback(self):
        input_dim = self.training_env.observation_space.shape[0] + self.training_env.action_space.shape[0]

        self.reward_model = RewardEnsemble(input_dim, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.segment_buffer = torch.empty((2, 0, self.segment_size, input_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)

        self.num_feed = 0


    def _on_training_start(self):
        self.annotation_buffer = ReplayBuffer(
            self.locals['total_timesteps'],
            self.training_env.observation_space,
            self.training_env.action_space,
            n_envs=self.training_env.num_envs
        )


    def _sample_segments(self, num_samples: int):
        start_indices = torch.randint(0, self.annotation_buffer.pos - self.segment_size, (num_samples,))

        offsets = torch.arange(0, self.segment_size).expand((num_samples, -1))
        step_indices = einops.repeat(start_indices, 'n -> n s', s=self.segment_size) + offsets

        env_indices = torch.randint(0, self.annotation_buffer.n_envs, (num_samples,))
        env_indices = einops.repeat(env_indices, 'n -> n s', s=self.segment_size)

        obs = torch.tensor(self.annotation_buffer.observations[step_indices, env_indices], dtype=torch.float32)
        act = torch.tensor(self.annotation_buffer.actions[step_indices, env_indices], dtype=torch.float32)
        gt_rewards = torch.tensor(self.annotation_buffer.rewards[step_indices, env_indices], dtype=torch.float32)

        return torch.cat([obs, act], dim=-1), gt_rewards


    def _query_segments(self, left: torch.Tensor, right: torch.Tensor, left_rew: torch.Tensor, right_rew: torch.Tensor):
        return left_rew.sum(dim=-1) > right_rew.sum(dim=-1)


    def _expand_data(self):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        segments, gt_rewards = self._sample_segments(num_samples)

        left, right = einops.rearrange(segments, '(n1 n2) s d -> n1 n2 s d', n1=2)
        left_rew, right_rew = einops.rearrange(gt_rewards, '(n1 n2) s -> n1 n2 s', n1=2)

        preferences = self._query_segments(left, right, left_rew, right_rew)

        self.segment_buffer = torch.cat([self.segment_buffer, torch.stack([left, right])], dim=1)
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])

        self.num_feed += num_samples


    def _train_reward_model(self):
        self.reward_model.train()

        for _ in range(self.n_epochs_reward):
            self.rew_optimizer.zero_grad()

            pred_rewards = self.reward_model(self.segment_buffer)
            pred_returns = einops.reduce(pred_rewards, 'n b s -> b n', 'sum')

            loss = self.rew_loss(pred_returns, self.preference_buffer)
            loss.backward()

            self.rew_optimizer.step()


    def _on_step(self):
        self.reward_model.eval()

        if self.num_timesteps % self.n_steps_reward == 0 and self.num_feed < self.max_feed:
            self._expand_data()
            self._train_reward_model()

        obs = self.model._last_obs
        act = self.locals['actions']

        self.annotation_buffer.add(obs, self.locals['new_obs'], act, self.locals['rewards'], self.locals['dones'], self.locals['infos'])

        state_actions = torch.cat([torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)], dim=-1)

        with torch.no_grad():
            pred_rewards = self.reward_model(state_actions)
            self.locals['rewards'] = pred_rewards.numpy()

        return True


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
