import einops
from hydra.utils import to_absolute_path
import itertools
import numpy as np
from pathlib import Path
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
        validate_on_train: bool = False,
        validate_on_held_out: bool = True,
        held_out_data_path: str = None,
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
        self.validate_on_train = validate_on_train
        self.validate_on_held_out = validate_on_held_out

        self.held_out_data_path = (
            Path(to_absolute_path(held_out_data_path))
            if held_out_data_path else None
        )


    def _init_callback(self):
        super()._init_callback()

        self.ensemble_reward_buffer = [[] for _ in range(self.training_env.num_envs)]

        obs_size, act_size = self._get_input_sizes()
        self.reward_model = RewardModel(obs_size + act_size, **self.reward_model_kwargs).to(self.device)
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.train_members_sequential:
            optim_params = [member.parameters() for member in self.reward_model.members]
        else:
            optim_params = [self.reward_model.parameters()]

        self.rew_optimizer = torch.optim.Adam(
            [{'params': params} for params in optim_params],
            lr=self.lr_reward
        )

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


    def _train_module_epoch(self, module: nn.Module, module_is_ensemble: bool = False):
        dataset = TensorDataset(self.segment_buffer[:self.num_feed], self.preference_buffer[:self.num_feed])
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        losses = []
        accuracies = []

        for segments, preferences in dataloader:
            self.rew_optimizer.zero_grad()

            loss, accuracy = self._compute_loss_for_module(module, segments, preferences, module_is_ensemble)
            loss.backward()
            self.rew_optimizer.step()

            losses.append(loss.item())
            accuracies.append(accuracy)

        return np.mean(losses), np.mean(accuracies)


    def _train_reward_model_epoch(self):
        if self.train_members_sequential:
            losses, accuracies = zip(*[self._train_module_epoch(member) for member in self.reward_model.members])
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
            if self.validate_on_train:
                self._validate_train()
            if self.validate_on_held_out:
                self._validate_held_out()


    def _calculate_corr_coef(self, pred_rewards: torch.Tensor, gt_rewards: torch.Tensor):
        gt_returns = einops.reduce(gt_rewards, 'pair batch segment -> (pair batch)', 'sum')
        pred_returns = einops.reduce(pred_rewards, 'pair batch segment -> (pair batch)', 'sum')
        stacked = torch.stack([gt_returns, pred_returns])
        corr_matrix = torch.corrcoef(stacked)
        return corr_matrix[0, 1]


    def _validate_on_segments(self, segments: torch.Tensor, rewards: torch.Tensor, preferences: torch.Tensor, compute_correlation: bool = False):
        self.reward_model.eval()

        dataset = TensorDataset(segments, preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, pin_memory=self.device.type == 'cuda')
        batch_statistics = [self._compute_loss_for_module(member, *batch) for batch in dataloader for member in self.reward_model.members]
        batch_losses, batch_accuracies = zip(*batch_statistics)

        batch_corrs = []
        if compute_correlation:
            rewards = einops.rearrange(rewards, 'pair batch segment -> batch pair segment')
            dataset = TensorDataset(segments, rewards)
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


    def _validate_on_episodes(self, episodes, size: int, compute_correlation: bool = False):
        segments, rewards = self.sampler.sample_segments(episodes, size, 'uniform', self.reward_model)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)
        return self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences, compute_correlation)


    def _log_validation_metrics(self, loss: float, accuracy: float, corr_coef: float | None, prefix: str = ''):
        self.logger.record(f"reward_model/eval/{prefix}loss", loss)
        self.logger.record(f"reward_model/eval/{prefix}accuracy", accuracy)
        if corr_coef:
            self.logger.record(f"reward_model/eval/{prefix}corr_coef", corr_coef)


    def _validate_train(self):
        loss, accuracy, corr_coef = self._validate_on_episodes(self.buffer.get_episodes(), len(self.segment_buffer))
        self._log_validation_metrics(loss, accuracy, corr_coef)


    def _validate_current(self):
        episodes = self.buffer.get_episodes()
        total_lens = itertools.accumulate(len(ep) for ep in reversed(episodes))
        eps_since_train = [ep for ep, total in zip(reversed(episodes), total_lens) if total <= self.n_steps_reward]

        loss, accuracy, corr_coef = self._validate_on_episodes(eps_since_train, int(0.5 * sum(len(ep) for ep in eps_since_train) / self.segment_size))
        self._log_validation_metrics(loss, accuracy, corr_coef, prefix='current/')


    def _validate_held_out(self):
        path = self.held_out_data_path
        if not path or not path.is_file():
            raise ValueError("""
                To validate on held-out data, please specify a path to the file containing the held-out data, relative to the current working directory.
                See notebooks/create_rm_validation_data.ipynb for the kind of file to point to.
            """)
        segments, rewards = torch.load(path)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)

        loss, accuracy, corr_coef = self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences)
        self._log_validation_metrics(loss, accuracy, corr_coef, prefix='held_out/')


    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1).to(self.device)

        with torch.no_grad():
            ensemble_rewards = self.reward_model(state_actions)
            for env_idx in range(len(ensemble_rewards[0])):
                self.ensemble_reward_buffer[env_idx].append(
                    ensemble_rewards[:, env_idx, 0].cpu().numpy()
                )
            return ensemble_rewards.mean(dim=0)


    def _save_returns(self, infos: list):
        super()._save_returns(infos)

        for env_idx, info in enumerate(infos):
            ep_info = info.get('episode')
            if ep_info is None:
                continue

            if self.ensemble_reward_buffer[env_idx]:
                ensemble_preds = np.array(self.ensemble_reward_buffer[env_idx])
                ep_info['pred_r_uncertainty'] = np.mean(np.std(ensemble_preds, axis=1))
                ep_info['pred_r_std_member'] = np.mean(np.std(ensemble_preds, axis=0))

                means = np.mean(ensemble_preds, axis=1)
                ep_info['pred_r_mean'] = np.mean(means)

                abs_means = np.abs(means) + 1e-8
                coef_vars = np.std(ensemble_preds, axis=1) / abs_means
                ep_info['pred_r_uncertainty_coef_var'] = np.mean(coef_vars)
                
                self.ensemble_reward_buffer[env_idx] = []


    def _on_step(self):
        continue_training = super()._on_step()
        if not continue_training: return False

        checkpoint_reached = self.num_timesteps % self.n_steps_eval_current == 0
        if self.has_trained and checkpoint_reached:
            with torch.no_grad():
                self._validate_current()
            self.logger.dump(step=self.num_timesteps)

        return True
