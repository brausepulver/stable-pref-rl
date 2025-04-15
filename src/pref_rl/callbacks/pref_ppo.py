import einops
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .pref import BasePrefCallback
from ..utils.model import build_layered_module
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
        n_epochs_reward: int = 100,
        train_acc_threshold_reward: float = 0.97,
        learning_rate_reward: float = 3e-4,
        batch_size_reward: int = 128,
        reward_model_kwargs: dict = {},
        n_steps_eval_current: int = None,
        train_members_sequential: bool = True,  # For consistency with B-Pref
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward
        self.n_steps_eval_current = n_steps_eval_current or self.n_steps_reward
        self.train_members_sequential = train_members_sequential


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()
        self.reward_model = RewardModel(obs_size + act_size, **self.reward_model_kwargs).to(self.device)

        self.is_equal_teacher = self.train_teacher.eps_equal > 0
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.train_members_sequential:
            self.rew_optimizer = [
                torch.optim.Adam(member.parameters(), lr=self.lr_reward)
                for member in self.reward_model.members
            ]
        else:
            self.rew_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr_reward)

        self.eval_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher='oracle')


    def _get_predictor(self):
        return self.reward_model


    def _calculate_accuracy(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        pred_preferences = torch.argmax(pred_returns, dim=-1)
        correct = (pred_preferences == preferences).float()
        return correct.mean().item()


    def _compute_loss_for_module(self, module: nn.Module, segments: torch.Tensor, preferences: torch.Tensor, module_is_ensemble: bool = False):
        segments = segments.to(self.device)
        preferences = preferences.to(self.device)
        pred_rewards = module(segments)

        pred_returns = einops.reduce(pred_rewards, '... batch pair segment 1 -> ... batch pair', 'sum')
        if module_is_ensemble:
            pred_returns = pred_returns.mean(dim=0)

        inputs = pred_returns[..., 1] - pred_returns[..., 0]
        loss = self.bce_loss(inputs, preferences)

        accuracy = self._calculate_accuracy(pred_returns, preferences)
        return loss, accuracy


    def _train_module_epoch(self, module: nn.Module, optim: torch.optim.Optimizer, module_is_ensemble: bool = False):
        dataset = TensorDataset(self.segment_buffer[:self.num_feed], self.preference_buffer[:self.num_feed])
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        losses = []
        accuracies = []

        for segments, preferences in dataloader:
            optim.zero_grad()

            loss, accuracy = self._compute_loss_for_module(module, segments, preferences, module_is_ensemble)
            loss.backward()
            optim.step()

            losses.append(loss.item())
            accuracies.append(accuracy)

        return np.mean(losses), np.mean(accuracies)


    def _train_reward_model_epoch(self):
        if self.train_members_sequential:
            losses, accuracies = zip(*[
                self._train_module_epoch(member, self.rew_optimizer[idx])
                for idx, member in enumerate(self.reward_model.members)
            ])
            return np.mean(losses), np.mean(accuracies)

        return self._train_module_epoch(self.reward_model, self.rew_optimizer, module_is_ensemble=True)


    def _train_predictor(self):
        self.reward_model.train()

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
            self._validate_total()


    def _calculate_corr_coef(self, pred_rewards: torch.Tensor, gt_rewards: torch.Tensor):
        gt_returns = einops.reduce(gt_rewards, 'pair batch segment -> (pair batch)', 'sum')
        pred_returns = einops.reduce(pred_rewards, 'pair batch segment -> (pair batch)', 'sum')
        stacked = torch.stack([gt_returns, pred_returns])
        corr_matrix = torch.corrcoef(stacked)
        return corr_matrix[0, 1]


    def _validate_on_episodes(self, episodes, size: int, compute_correlation: bool = True):
        self.reward_model.eval()

        segments, rewards = self.sampler.sample_segments(episodes, size, 'uniform', self.reward_model)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)

        dataset = TensorDataset(segments[keep_indices], preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, pin_memory=self.device.type == 'cuda')
        batch_statistics = [self._compute_loss_for_module(member, *batch) for batch in dataloader for member in self.reward_model.members]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        batch_corrs = []
        if compute_correlation:
            rewards = einops.rearrange(rewards, 'pair batch segment -> batch pair segment')
            dataset = TensorDataset(segments[keep_indices], rewards[keep_indices])
            dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, pin_memory=self.device.type == 'cuda')

            for segments, rewards in dataloader:
                pred_rewards = self.reward_model(segments.to(self.device)).mean(dim=0).squeeze(-1)
                corr_coef = self._calculate_corr_coef(pred_rewards, rewards.to(self.device))
                batch_corrs.append(corr_coef)

        return (
            torch.tensor(batch_losses).mean().item(),
            torch.tensor(batch_accuracies).mean().item(),
            torch.tensor(batch_corrs).mean().item() if compute_correlation else None,
        )


    def _validate_total(self):
        loss_total, accuracy_total, corr_coef = self._validate_on_episodes(self.buffer.episodes, len(self.segment_buffer))
        self.logger.record('reward_model/eval/loss', loss_total)
        self.logger.record('reward_model/eval/accuracy', accuracy_total)
        self.logger.record('reward_model/eval/corr_coef', corr_coef)


    def _validate_current(self):
        total_lens = itertools.accumulate(len(ep) for ep in reversed(self.buffer.episodes))
        recent_eps = [ep for ep, total in zip(reversed(self.buffer.episodes), total_lens) if total <= self.n_steps_reward]

        loss_current, accuracy_current, corr_coef = self._validate_on_episodes(recent_eps, int(0.5 * sum(len(ep) for ep in recent_eps) / self.segment_size))
        self.logger.record('reward_model/eval/loss_current', loss_current)
        self.logger.record('reward_model/eval/accuracy_current', accuracy_current)
        self.logger.record('reward_model/eval/corr_coef_current', corr_coef)


    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1).to(self.device)

        with torch.no_grad():
            return self.reward_model(state_actions).mean(dim=0).detach()


    def _on_step(self):
        continue_training = super()._on_step()
        if not continue_training: return False

        checkpoint_reached = self.num_timesteps % self.n_steps_eval_current == 0
        if self.has_trained and checkpoint_reached:
            with torch.no_grad():
                self._validate_current()
            self.logger.dump(step=self.num_timesteps)

        return True
