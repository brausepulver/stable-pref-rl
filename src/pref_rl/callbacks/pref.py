import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from typing_extensions import override
from .reward_mod import RewardModifierCallback
from ..utils.model import build_layered_module
from ..utils.pref import EpisodeBuffer, Teacher, Sampler


class RewardModel(nn.Module):
    def __init__(self, input_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([self._build_member(input_dim, **kwargs) for _ in range(ensemble_size)])


    def _build_member(self, input_dim, net_arch=[256, 256, 256], activation_fn=nn.LeakyReLU, output_fn=nn.Tanh):
        return build_layered_module(input_dim, net_arch, activation_fn, output_fn)


    def forward(self, x):
        return torch.stack([member(x) for member in self.members])


class PrefCallback(RewardModifierCallback):
    def __init__(self,
        n_steps_reward: int = 32_000,
        ann_buffer_size_eps: int = 100,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_batch_size: int = 200,
        n_epochs_reward: int = 100,
        train_acc_threshold_reward: float = 0.97,
        learning_rate_reward: float = 3e-4,
        batch_size_reward: int = 128,
        reward_model_kwargs: dict = {},
        teacher: str = None,
        teacher_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(log_prefix='reward_model/', **kwargs)

        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.sampling_method = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward

        self.teacher = teacher
        self.teacher_kwargs = teacher_kwargs
        self.num_feed = 0


    def _init_callback(self):
        self.observation_size = self.training_env.observation_space.shape[0]
        self.action_size = self.training_env.action_space.shape[0]
        input_dim = self.observation_size + self.action_size

        self.reward_model = RewardModel(input_dim, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.segment_buffer = torch.empty((0, 2, self.segment_size, input_dim))
        self.preference_buffer = torch.empty((0,), dtype=torch.long)

        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=self.observation_size, action_size=self.action_size)
        self.teacher = Teacher(segment_size=self.segment_size, teacher=self.teacher, teacher_kwargs=self.teacher_kwargs)


    def _calculate_accuracy(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        pred_preferences = torch.argmax(pred_returns.mean(dim=0), dim=-1)
        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _compute_reward_model_loss(self, segments: torch.Tensor, preferences: torch.Tensor):
        pred_rewards = self.reward_model(segments)
        pred_returns = einops.reduce(pred_rewards, 'm b n s 1 -> m b n', 'sum')

        loss = torch.stack([self.rew_loss(member_returns, preferences) for member_returns in pred_returns]).sum()
        accuracy = self._calculate_accuracy(pred_returns, preferences)
        return loss, accuracy


    def _expand_data(self):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.episodes, self.reward_model, num_samples, self.sampling_method)

        preferences, keep_indices = self.teacher.query_segments(rewards)

        self.segment_buffer = torch.cat([self.segment_buffer, state_actions[keep_indices]])
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])

        self.num_feed += len(keep_indices)
        self.logger.record('reward_model/num_feed', self.num_feed)


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

        segments, rewards = self.sampler.sample_segments(self.buffer.episodes, self.reward_model, len(self.segment_buffer), 'uniform')
        preferences, keep_indices = self.teacher.query_segments(rewards)

        dataset = TensorDataset(segments[keep_indices], preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True)

        batch_statistics = [self._compute_reward_model_loss(*batch) for batch in dataloader]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        self.logger.record('reward_model/eval/loss', np.mean(batch_losses))
        self.logger.record('reward_model/eval/accuracy', np.mean(batch_accuracies))


    def _get_current_step(self):
        obs = torch.tensor(self.model._last_obs, dtype=torch.float)
        act = torch.tensor(self.locals['actions'], dtype=torch.float)
        gt_rewards = torch.tensor(self.locals['rewards'], dtype=torch.float).unsqueeze(-1)
        return obs, act, gt_rewards


    @override
    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1)

        with torch.no_grad():
            return self.reward_model(state_actions).mean(dim=0)


    def _on_step(self):
        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        if self.locals['dones'].any():
            self.teacher.update_thresholds(self.buffer.episodes)

        checkpoint_reached = self.num_timesteps % self.n_steps_reward == 0
        feedback_left = self.num_feed < self.max_feed
        buffer_empty = len(self.buffer.episodes) == 0

        if checkpoint_reached and feedback_left and not buffer_empty:
            self._expand_data()
            self._train_reward_model()
            with torch.no_grad():
                self._validate_reward_model()
            self._on_event()

        return super()._on_step()
