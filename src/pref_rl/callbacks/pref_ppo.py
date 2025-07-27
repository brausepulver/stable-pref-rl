from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import einops
from hydra.utils import to_absolute_path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset

from .pref import BasePrefCallback
from ..utils.data import MaskedDataset
from ..utils.reward_models import RewardModel
from ..utils.sampler import NoValidEpisodesError
from ..utils.schedules import PrefPPOScheduleState
from ..utils.teacher import Teacher


class PrefPPOCallback(BasePrefCallback):
    def __init__(self,
        n_epochs_reward: int = 100,
        train_acc_threshold_reward: float = 0.97,
        learning_rate_reward: float = 3e-4,
        batch_size_reward: int = 128,
        reward_model_kwargs: dict = {},
        n_steps_eval_current: int | None = None,
        validate_on_train: bool = False,
        validate_on_current: bool = True,
        validate_on_held_out: bool = True,
        held_out_data_path: str | None = None,
        log_sampler_metrics: bool = True,
        ensemble_agg_fn: Callable = lambda pred: pred.mean(dim=0),
        reward_model_kind: str = 'reward_model',
        **kwargs
    ):
        super().__init__(log_sampler_metrics=log_sampler_metrics, **kwargs)

        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward
        self.n_steps_eval_current = n_steps_eval_current or self.schedule.n_steps_reward
        self.validate_on_train = validate_on_train
        self.validate_on_current = validate_on_current
        self.validate_on_held_out = validate_on_held_out
        self.log_sampler_metrics = log_sampler_metrics
        self.ensemble_agg_fn = ensemble_agg_fn

        self.reward_model_cls = {
            'reward_model': RewardModel,
        }[reward_model_kind]

        self.held_out_data_path = (
            Path(to_absolute_path(held_out_data_path))
            if held_out_data_path else None
        )


    def _create_schedule_state(self):
        base_state = super()._create_schedule_state()
        return PrefPPOScheduleState(
            num_timesteps=base_state.num_timesteps,
            total_timesteps=base_state.total_timesteps,
            training_progress=base_state.training_progress,
            progress_remaining=base_state.progress_remaining,
            has_trained=base_state.has_trained,
            steps_since_train=base_state.steps_since_train,
            buffer=base_state.buffer,
            feed_buffer=base_state.feed_buffer,
            synth_buffer=base_state.synth_buffer,
            reward_model=self.reward_model
        )


    def _init_callback(self):
        super()._init_callback()

        self.ensemble_reward_buffer = [[] for _ in range(self.training_env.num_envs)]

        obs_size, act_size = self._get_input_sizes()
        self.reward_model = self.reward_model_cls(obs_size + act_size, **self.reward_model_kwargs).to(self.device)

        self.rew_optimizers = [
            torch.optim.Adam(member.parameters(), lr=self.lr_reward)
            for member in self.reward_model.members
        ]

        self.eval_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher='oracle')


    def _get_predictor(self):
        return self.reward_model


    def _compute_loss(self, pred_rewards: torch.Tensor, preferences: torch.Tensor, weight: Optional[torch.Tensor] = None):
        pred_returns = einops.reduce(pred_rewards, '... batch pair segment 1 -> ... batch pair', 'sum')
        inputs = pred_returns[..., 1] - pred_returns[..., 0]
        criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='none')
        losses = criterion(inputs, preferences)

        pred_preferences = torch.argmax(pred_returns, dim=-1)
        correct = (pred_preferences == preferences)

        return losses, correct


    def _compute_batch_metrics(self, losses: torch.Tensor, correct: torch.Tensor, mask: torch.Tensor):
        real_idx = torch.argwhere(mask == 0)
        synth_idx = torch.argwhere(mask == 1)
        
        metrics = {
            'loss': losses.mean().item(),
            'loss_correct': losses[correct].mean().item(),
            'loss_incorrect': losses[~correct].mean().item(),
            'accuracy': correct.mean(dtype=torch.float).item(),
            'real_loss': losses[real_idx].mean().item(),
            'real_acc': correct[real_idx].mean(dtype=torch.float).item(),
        }

        if synth_idx.numel():
            metrics |= {
                'synth_loss': losses[synth_idx].mean().item(),
                'synth_acc': correct[synth_idx].mean(dtype=torch.float).item(),
            }

        return metrics


    def _train_member_epoch(self, member: nn.Module, optimizer, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        metrics = []

        for segments, preferences, weights, mask in dataloader:
            optimizer.zero_grad()

            pred_rewards = member(segments)
            losses, correct = self._compute_loss(pred_rewards, preferences, weights)

            losses.sum().backward()
            optimizer.step()

            metrics.append(self._compute_batch_metrics(losses, correct, mask))

        return metrics


    def _get_dataset(self):
        real_dataset = MaskedDataset(self.feed_buffer.get_dataset(), 0)
        synth_dataset = MaskedDataset(self.synth_buffer.get_dataset(), 1)
        return ConcatDataset([real_dataset, synth_dataset])


    def _train_reward_model_epoch(self):
        dataset = self._get_dataset()
        metrics = []

        for index, member in enumerate(self.reward_model.members):
            optimizer = self.rew_optimizers[index]
            member_metrics = self._train_member_epoch(member, optimizer, dataset)
            metrics.append(member_metrics)

        return metrics


    def _average_metrics(self, metrics: list):
        grouped_metrics = defaultdict(list)
        for batch_metrics in metrics:
            for name, value in batch_metrics.items():
                grouped_metrics[name].append(value)

        avg_metrics = {key: np.mean(value) for key, value in grouped_metrics.items()}
        return avg_metrics


    def _train_predictor(self):
        self.reward_model.train()

        metrics = []
        for epoch in range(self.n_epochs_reward):
            epoch_metrics = self._train_reward_model_epoch()
            metrics.append(epoch_metrics)

            epoch_accuracy = np.mean([batch['accuracy'] for member in epoch_metrics for batch in member if 'accuracy' in batch])
            if epoch_accuracy > self.train_acc_threshold_reward:
                break

        flat_metrics = [batch for epoch in metrics for member in epoch for batch in member]
        avg_metrics = self._average_metrics(flat_metrics)
        log_metrics = avg_metrics | {'epochs': epoch + 1}
        self.logger.record_with_progress(log_metrics, self.num_feed, self.training_progress, prefix="reward_model/train/")

        with torch.no_grad():
            if self.validate_on_train:
                self._validate_train()
            if self.validate_on_held_out:
                self._validate_held_out()


    def _validate(self, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        metrics = []

        for segments, preferences, weights, mask in dataloader:
            pred_rewards = self.ensemble_agg_fn(self.reward_model(segments))
            losses, correct = self._compute_loss(pred_rewards, preferences, weights)
            metrics.append(self._compute_batch_metrics(losses, correct, mask))

        return metrics


    def _validate_on_episodes(self, episodes, episode_ages, size: int, compute_sampler_metrics: bool = True):
        schedule_state = self._create_schedule_state()
        segments, rewards, sampler_metrics = self.sampler.sample_pairs(episodes, episode_ages, size, reward_model=self.reward_model, compute_uniform_metrics=compute_sampler_metrics, schedule_state=schedule_state)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)

        dataset = TensorDataset(segments[keep_indices], preferences, torch.ones(len(preferences)), torch.zeros(len(preferences)))
        metrics = self._average_metrics(self._validate(dataset))

        return {**metrics, 'sampler_metrics': sampler_metrics}


    def _log_validation_metrics(self, metrics: dict, prefix: str = ''):
        if self.log_sampler_metrics and metrics.get('sampler_metrics') is not None:
            self._log_metrics_stats(metrics.pop('sampler_metrics'), prefix=prefix)

        self.logger.record_with_progress(metrics, self.num_feed, self.training_progress, prefix=f'reward_model/eval/{prefix}')


    def _validate_train(self):
        metrics = self._validate_on_episodes(self.buffer.get_episodes(), self.buffer.get_episode_ages(), len(self.feed_buffer))
        self._log_validation_metrics(metrics)


    def _validate_current(self):
        episodes = self.buffer.get_episodes()[-self.done_eps_since_eval:]
        episode_ages = self.buffer.get_episode_ages()[-self.done_eps_since_eval:]
        self.done_eps_since_eval = 0
        try:
            metrics = self._validate_on_episodes(episodes, episode_ages, int(0.5 * sum(len(ep) for ep in episodes) / self.segment_size))
            self._log_validation_metrics(metrics, prefix='current/')
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
        schedule_state = self._create_schedule_state()
        sampler_metrics = self.sampler.compute_logging_metrics(segments, self.reward_model, schedule_state=schedule_state)

        dataset = TensorDataset(segments[keep_indices], preferences, torch.ones(len(preferences)), torch.zeros(len(preferences)))
        metrics = self._average_metrics(self._validate(dataset))
        self._log_validation_metrics({**metrics, 'sampler_metrics': sampler_metrics}, prefix='held_out/')


    def _predict_rewards(self):
        self.reward_model.eval()

        obs, act, _ = self._get_current_step()
        state_actions = torch.cat([obs, act], dim=-1).to(self.device)

        with torch.no_grad():
            member_rewards = self.reward_model(state_actions)
            for env_idx in range(len(member_rewards[0])):
                self.ensemble_reward_buffer[env_idx].append(
                    member_rewards[:, env_idx, 0].cpu().numpy()
                )
            pred_rewards = self.ensemble_agg_fn(member_rewards)

        return pred_rewards


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
