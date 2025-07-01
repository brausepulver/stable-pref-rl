import einops
from hydra.utils import to_absolute_path
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .pref import BasePrefCallback
from ..utils.model import build_layered_module
from ..utils.pref import NoValidEpisodesError, Teacher


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
        validate_on_current: bool = True,
        validate_on_held_out: bool = True,
        held_out_data_path: str = None,
        log_sampler_metrics: bool = True,
        ensemble_disjoint_data: bool = False,
        **kwargs
    ):
        super().__init__(log_sampler_metrics=log_sampler_metrics, **kwargs)

        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward
        self.n_steps_eval_current = n_steps_eval_current or self.n_steps_reward
        self.train_members_sequential = train_members_sequential
        self.validate_on_train = validate_on_train
        self.validate_on_current = validate_on_current
        self.validate_on_held_out = validate_on_held_out
        self.log_sampler_metrics = log_sampler_metrics
        self.ensemble_disjoint_data = ensemble_disjoint_data

        self.held_out_data_path = (
            Path(to_absolute_path(held_out_data_path))
            if held_out_data_path else None
        )

        if self.run is not None:
            self.run.define_metric(step_metric='pref/training_progress', name='reward_model/*')
            self.run.define_metric(step_metric='pref/num_feed', name='reward_model/*')


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


    def _train_module_epoch(self, module: nn.Module, dataset: TensorDataset, module_is_ensemble: bool = False):
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
        segments = self.segment_buffer[:self.num_feed]
        preferences = self.preference_buffer[:self.num_feed]
        dataset = TensorDataset(segments, preferences)

        if not self.train_members_sequential:
            return self._train_module_epoch(self.reward_model, dataset, self.rew_optimizer, module_is_ensemble=True)

        losses = []
        accuracies = []

        for index, member in enumerate(self.reward_model.members):
            if self.ensemble_disjoint_data:
                step = len(self.reward_model.members)
                idx = torch.arange(index, self.num_feed, step, device=segments.device)
                dataset = TensorDataset(segments[idx], preferences[idx])
            
            loss, accuracy = self._train_module_epoch(member, dataset)
            losses.append(loss)
            accuracies.append(accuracy)

        return np.mean(losses), np.mean(accuracies)


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

        if self.run:
            self.run.log({
                'reward_model/train/loss': np.mean(losses),
                'reward_model/train/accuracy': np.mean(accuracies),
                'reward_model/train/epochs': epoch + 1,
                'pref/num_feed': self.num_feed,
                'pref/training_progress': self.training_progress,
            })

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


    def _validate_on_episodes(self, episodes, size: int, compute_correlation: bool = False, compute_sampler_metrics: bool = True):
        segments, rewards, sampler_metrics = self.sampler.sample_segments(episodes, size, 'uniform', self.reward_model, compute_uniform_metrics=compute_sampler_metrics)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)
        return (
            *self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences, compute_correlation),
            sampler_metrics
        )


    def _log_validation_metrics(self, loss: float, accuracy: float, corr_coef: float | None, sampler_metrics: dict | None, prefix: str = ''):
        self.logger.record(f"reward_model/eval/{prefix}loss", loss)
        self.logger.record(f"reward_model/eval/{prefix}accuracy", accuracy)
        if corr_coef:
            self.logger.record(f"reward_model/eval/{prefix}corr_coef", corr_coef)
        if self.log_sampler_metrics and sampler_metrics:
            self._log_sampler_metrics(sampler_metrics, prefix=prefix)

        if self.run:
            self.run.log({
                f'reward_model/eval/{prefix}loss': loss,
                f'reward_model/eval/{prefix}accuracy': accuracy,
                **({ f'reward_model/eval/{prefix}corr_coef': corr_coef } if corr_coef is not None else {}),
                'pref/num_feed': self.num_feed,
                'pref/training_progress': self.training_progress,
            })


    def _validate_train(self):
        loss, accuracy, corr_coef, sampler_metrics = self._validate_on_episodes(self.buffer.get_episodes(), len(self.segment_buffer))
        self._log_validation_metrics(loss, accuracy, corr_coef, sampler_metrics)


    def _validate_current(self):
        episodes = self.buffer.get_episodes()[-self.done_eps_since_eval:]
        self.done_eps_since_eval = 0
        try:
            loss, accuracy, corr_coef, sampler_metrics = self._validate_on_episodes(episodes, int(0.5 * sum(len(ep) for ep in episodes) / self.segment_size))
            self._log_validation_metrics(loss, accuracy, corr_coef, sampler_metrics, prefix='current/')
        except NoValidEpisodesError:
            return


    def _validate_held_out(self):
        path = self.held_out_data_path
        if not path or not path.is_file():
            raise ValueError("""
                To validate on held-out data, please specify a path to the file containing the held-out data, relative to the current working directory.
                See notebooks/create_rm_validation_data.ipynb for the kind of file to point to.
            """)
        segments, rewards = torch.load(path)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)
        sampler_metrics = self.sampler.compute_metrics(segments, self.reward_model)

        loss, accuracy, corr_coef = self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences)
        self._log_validation_metrics(loss, accuracy, corr_coef, sampler_metrics, prefix='held_out/')


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

                if self.run:
                    self.run.log({
                        'pref/ep_rew_mean': ep_info['pred_r_mean'],
                        'pref/step_rew_mean': ep_info['pred_r_step'],
                        'pref/ep_rew_std': ep_info['pred_r_std'],
                        'reward_model/avg_member_std_ep': ep_info['pred_r_std_member'],
                        'reward_model/avg_ensemble_std_rew': ep_info['pred_r_uncertainty'],
                        'pref/num_feed': self.num_feed,
                        'pref/training_progress': self.training_progress,
                    })


    def _on_training_start(self) -> None:
        self.done_eps_since_eval = 0


    def _on_step(self):
        self.done_eps_since_eval += self.locals['dones'].sum().item()

        continue_training = super()._on_step()
        if not continue_training: return False

        checkpoint_reached = self.num_timesteps % self.n_steps_eval_current == 0
        if self.has_trained and checkpoint_reached:
            if self.validate_on_current:
                with torch.no_grad():
                    self._validate_current()
                self.logger.dump(step=self.num_timesteps)

        return True
