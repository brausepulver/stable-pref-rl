import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from ..utils.model import build_layered_module
from .pref import BasePrefCallback
from ..utils.pref import Teacher


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
        device: str = 'cpu',
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

        self.device = torch.device(
            device or
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()
        self.reward_model = RewardModel(obs_size + act_size, **self.reward_model_kwargs)
        self.rew_loss = nn.CrossEntropyLoss()
        self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.eval_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher='oracle')


    def _get_predictor(self):
        return self.reward_model


    def _calculate_accuracy(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        pred_preferences = torch.argmax(pred_returns, dim=-1)
        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _compute_loss_for_member(self, member: nn.Module, segments: torch.Tensor, preferences: torch.Tensor):
        segments = segments.to(self.device)
        preferences = preferences.to(self.device)
        pred_rewards = member(segments)
        pred_returns = einops.reduce(pred_rewards, 'b n s 1 -> b n', 'sum')
        loss = self.rew_loss(pred_returns, preferences)
        accuracy = self._calculate_accuracy(pred_returns, preferences)
        return loss, accuracy


    def _train_member_epoch(self, member: nn.Module):
        dataset = TensorDataset(self.segment_buffer, self.preference_buffer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        losses = []
        accuracies = []

        for segments, preferences in dataloader:
            self.rew_optimizer.zero_grad()
            loss, accuracy = self._compute_loss_for_member(member, segments, preferences)
            loss.backward()
            self.rew_optimizer.step()

            losses.append(loss.item())
            accuracies.append(accuracy)

        return np.mean(losses), np.mean(accuracies)


    def _train_reward_model_epoch(self):
        losses, accuracies = zip(*[self._train_member_epoch(member) for member in self.reward_model.members])
        return np.mean(losses), np.mean(accuracies)


    def _train_predictor(self):
        self.reward_model.train()
        self.reward_model.to(self.device)

        losses = []
        accuracies = []

        for epoch in range(self.n_epochs_reward):
            loss, accuracy = self._train_reward_model_epoch()
            losses.append(loss)
            accuracies.append(accuracy)

            if accuracy > self.train_acc_threshold_reward:
                break

        self.logger.record('reward_model/train/loss', np.mean(losses))
        self.logger.record('reward_model/train/accuracy', np.mean(accuracies))
        self.logger.record('reward_model/train/epochs', epoch + 1)

        with torch.no_grad():
            self._validate_reward_model()

        self.reward_model.to(self.model.device)


    def _validate_reward_model(self):
        self.reward_model.eval()

        segments, rewards = self.sampler.sample_segments(self.buffer.episodes, len(self.segment_buffer), 'uniform', self.reward_model)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)

        dataset = TensorDataset(segments[keep_indices], preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')

        batch_statistics = [self._compute_loss_for_member(member, *batch) for batch in dataloader for member in self.reward_model.members]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        self.logger.record('reward_model/eval/loss', torch.tensor(batch_losses).mean().item())
        self.logger.record('reward_model/eval/accuracy', torch.tensor(batch_accuracies).mean().item())


    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1)

        with torch.no_grad():
            return self.reward_model(state_actions).mean(dim=0)
