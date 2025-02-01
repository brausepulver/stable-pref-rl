from collections import deque
import einops
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace


class RewardModel(nn.Module):
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
        out = torch.cat([member(x) for member in self.members], dim=-1)
        return SimpleNamespace(value=out.mean(dim=-1), uncertainty=out.std(dim=-1))


class EpisodeBuffer():
    def __init__(self, num_envs, num_episodes):
        self._env_buffer = [[] for _ in range(num_envs)]
        self.episodes = deque(maxlen=num_episodes)


    def add(self, value: torch.Tensor, done: np.ndarray):
        for env_idx, env_value in enumerate(value):
            self._env_buffer[env_idx].append(env_value)

        for env_idx in np.argwhere(done).squeeze():
            episode = self._env_buffer[env_idx]
            self.episodes.append(torch.stack(episode))
            episode.clear()


class PrefCallback(BaseCallback):
    def __init__(self,
        n_steps_reward: int = 32_000,
        ann_buffer_size: int = 100,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_batch_size: int = 200,
        n_epochs_reward: int = 100,
        train_acc_threshold_reward: float = 0.97,
        learning_rate_reward: float = 3e-4,
        batch_size_reward: int = 128,
        reward_model_kwargs: dict = {},
        teacher: dict = {'beta': -1, 'gamma': 1, 'eps_mistake': 0, 'eps_skip': 0, 'eps_equal': 0},
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size = ann_buffer_size
        self.sampler = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward

        self.pre_sample_multiplier = 10


    def _init_callback(self):
        self.annotation_buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size)

        self.observation_size = self.training_env.observation_space.shape[0]
        self.action_size = self.training_env.action_space.shape[0]

        input_dim = self.observation_size + self.action_size

        self.reward_model = RewardModel(input_dim, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.segment_buffer = torch.empty((0, 2, self.segment_size, input_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)

        self.num_feed = 0


    def _on_rollout_start(self):
        self.pred_reward_buffer = [[] for _ in range(self.training_env.num_envs)]


    def _sample_segments(self, num_samples: int, sampler: str):
        episodes = self.annotation_buffer.episodes
        valid_episodes = [ep for ep in episodes if len(ep) >= self.segment_size]

        num_samples_expanded = num_samples if sampler == 'uniform' else self.pre_sample_multiplier * self.feed_batch_size
        ep_indices = torch.randint(0, len(valid_episodes), (2 * num_samples_expanded,))

        segments = []

        for ep_idx in ep_indices:
            ep = valid_episodes[ep_idx]

            start_step = np.random.randint(0, len(ep) - self.segment_size)
            offsets = torch.arange(0, self.segment_size)

            step_indices = start_step + offsets
            segment = ep[step_indices]

            segments.append(segment)

        obs, act, gt_rewards = torch.split(torch.stack(segments), (self.observation_size, self.action_size, 1), dim=-1)
        state_actions = torch.cat([obs, act], dim=-1)

        if sampler == 'uniform':
            return state_actions, gt_rewards.squeeze()

        elif sampler == 'disagreement':
            with torch.no_grad():
                uncertainties = self.reward_model(state_actions).uncertainty
                largest_std_indices = torch.topk(uncertainties.sum(dim=-1), num_samples, dim=0).indices

                return state_actions[largest_std_indices], gt_rewards[largest_std_indices].squeeze()


    def _query_segments(self, left: torch.Tensor, right: torch.Tensor, left_rew: torch.Tensor, right_rew: torch.Tensor):
        return (left_rew.sum(dim=-1) < right_rew.sum(dim=-1)).to(dtype=torch.long)


    def _generate_data(self, num_samples: int, sampler: str):
        segments, gt_rewards = self._sample_segments(num_samples, sampler)

        left, right = einops.rearrange(segments, '(n1 n2) s d -> n1 n2 s d', n1=2)
        left_rew, right_rew = einops.rearrange(gt_rewards, '(n1 n2) s -> n1 n2 s', n1=2)

        segments = torch.stack([left, right], dim=1)
        preferences = self._query_segments(left, right, left_rew, right_rew)

        return segments, preferences


    def _expand_data(self):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        segments, preferences = self._generate_data(num_samples, self.sampler)

        self.segment_buffer = torch.cat([self.segment_buffer, segments])
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])

        self.num_feed += num_samples
        self.logger.record('reward_model/num_feed', self.num_feed)


    def _calculate_accuracy(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        pred_preferences = torch.argmax(pred_returns, dim=1)

        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _compute_reward_model_loss(self, segments: torch.Tensor, preferences: torch.Tensor):
        pred_rewards = self.reward_model(segments).value
        pred_returns = einops.reduce(pred_rewards, 'b n s -> b n', 'sum')

        loss = self.rew_loss(pred_returns, preferences)
        accuracy = self._calculate_accuracy(pred_returns, preferences)

        return loss, accuracy


    def _train_reward_model(self):
        self.reward_model.train()

        dataset = TensorDataset(self.segment_buffer, self.preference_buffer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True)

        batch_losses = []
        batch_accuracies = []

        for epoch in range(self.n_epochs_reward):
            batch_acc_epoch = []

            for segments, preferences in dataloader:
                self.rew_optimizer.zero_grad()

                loss, accuracy = self._compute_reward_model_loss(segments, preferences)
                loss.backward()
                self.rew_optimizer.step()

                batch_losses.append(loss.item())
                batch_acc_epoch.append(accuracy)

            batch_accuracies.extend(batch_acc_epoch)

            if np.mean(batch_acc_epoch) > self.train_acc_threshold_reward:
                break

        self.logger.record('reward_model/train/loss', np.mean(batch_losses))
        self.logger.record('reward_model/train/accuracy', np.mean(batch_accuracies))
        self.logger.record('reward_model/train/epochs', epoch + 1)


    def _validate_reward_model(self):
        self.reward_model.eval()

        data = self._generate_data(len(self.segment_buffer), 'uniform')
        dataset = TensorDataset(*data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True)

        batch_statistics = [self._compute_reward_model_loss(*batch) for batch in dataloader]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        self.logger.record('reward_model/eval/loss', np.mean(batch_losses))
        self.logger.record('reward_model/eval/accuracy', np.mean(batch_accuracies))


    def _predict_rewards(self, state_actions: torch.Tensor):
        pred_rewards = self.reward_model(state_actions).value

        for env_idx in range(len(pred_rewards)):
            pred_reward = pred_rewards[env_idx].item()

            self.locals['rewards'][env_idx] = pred_reward  # We cannot modify the reference in self.locals['rewards'] directly
            self.pred_reward_buffer[env_idx].append(pred_reward)


    def _log_pred_reward(self, infos):
        for env_idx, info in enumerate(infos):
            ep_info = info.get('episode')
            if ep_info is None:
                continue

            ep_mean_pred_r = np.mean(self.pred_reward_buffer[env_idx])
            ep_info['pred_r'] = ep_mean_pred_r
            self.pred_reward_buffer[env_idx] = []


    def _on_step(self):
        self.reward_model.eval()

        obs = torch.tensor(self.model._last_obs, dtype=torch.float32)
        act = torch.tensor(self.locals['actions'], dtype=torch.float32)

        state_actions = torch.cat([obs, act], dim=-1)

        gt_rewards = torch.tensor(self.locals['rewards'], dtype=torch.float32).unsqueeze(-1)
        annotations = torch.cat([state_actions, gt_rewards], dim=-1)

        self.annotation_buffer.add(annotations, self.locals['dones'])

        checkpoint_reached = self.num_timesteps % self.n_steps_reward == 0
        feedback_left = self.num_feed < self.max_feed
        buffer_empty = len(self.annotation_buffer.episodes) == 0

        if checkpoint_reached and feedback_left and not buffer_empty:
            self._expand_data()
            self._train_reward_model()

            with torch.no_grad():
                self._validate_reward_model()

        with torch.no_grad():
            self._predict_rewards(state_actions)

        self._log_pred_reward(self.locals['infos'])

        return True


    def _on_rollout_end(self):
        ep_pred_rew = [ep_info['pred_r'] for ep_info in self.model.ep_info_buffer if 'pred_r' in ep_info]

        if ep_pred_rew:
            self.logger.record('reward_model/ep_pred_rew_mean', np.mean(ep_pred_rew))
