import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from ..utils.model import build_layered_module
from .pref import BasePrefCallback


class RewardModel(nn.Module):
    def __init__(self, input_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([self._build_member(input_dim, **kwargs) for _ in range(ensemble_size)])


    def _build_member(self, input_dim, net_arch=[256, 256, 256], activation_fn=nn.LeakyReLU(), output_fn=nn.Tanh()):
        return build_layered_module(input_dim, net_arch, activation_fn, output_fn)


    def forward(self, x):
        return torch.stack([member(x) for member in self.members])


class PrefPPOCallback(BasePrefCallback):
    def __init__(self,
        n_epochs_reward: int = 100,
        train_acc_threshold_reward: float = 0.97,
        learning_rate_reward: float = 3e-4,
        batch_size_reward: int = 128,
        reward_model_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward


    def _init_callback(self):
        super()._init_callback()

        input_dim = self.observation_size + self.action_size
        self.reward_model = RewardModel(input_dim, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)


    def _calculate_accuracy(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        pred_preferences = torch.argmax(pred_returns, dim=-1)
        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _compute_reward_model_loss(self, segments: torch.Tensor, preferences: torch.Tensor):
        pred_rewards = self.reward_model(segments)
        pred_returns = einops.reduce(pred_rewards, 'm b n s 1 -> m b n', 'sum')

        loss = torch.stack([self.rew_loss(member_returns, preferences) for member_returns in pred_returns]).sum()
        accuracy = self._calculate_accuracy(pred_returns, preferences)
        return loss, accuracy


    def _get_predictor(self):
        return self.reward_model


    def _train_predictor(self):
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

        with torch.no_grad():
            self._validate_predictor()


    def _validate_predictor(self):
        self.reward_model.eval()

        segments, rewards = self.sampler.sample_segments(self.buffer.episodes, len(self.segment_buffer), 'uniform', self.reward_model)
        preferences, keep_indices = self.teacher.query_segments(rewards)

        dataset = TensorDataset(segments[keep_indices], preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True)

        batch_statistics = [self._compute_reward_model_loss(*batch) for batch in dataloader]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        self.logger.record('reward_model/eval/loss', np.mean(batch_losses))
        self.logger.record('reward_model/eval/accuracy', np.mean(batch_accuracies))


    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1)

        with torch.no_grad():
            return self.reward_model(state_actions).mean(dim=0)
