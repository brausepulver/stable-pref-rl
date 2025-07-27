from collections import defaultdict
import einops
from hydra.utils import to_absolute_path
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, ConcatDataset
from typing import Callable, Optional

from .pref import BasePrefCallback
from ..utils.data import MaskedDataset
from ..utils.reward_models import RewardModel, MultiHeadMember, MultiHeadRewardModel
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
        train_members_sequential: bool = True,  # For consistency with B-Pref
        validate_on_train: bool = False,
        validate_on_current: bool = True,
        validate_on_held_out: bool = True,
        held_out_data_path: str | None = None,
        log_sampler_metrics: bool = True,
        ensemble_disjoint_data: bool = False,
        ensemble_agg_fn: Callable = lambda pred: pred.mean(dim=0),
        reward_model_kind: str = 'reward_model',
        num_samples_ep_age: Optional[int] = None,
        **kwargs
    ):
        super().__init__(log_sampler_metrics=log_sampler_metrics, **kwargs)

        self.n_epochs_reward = n_epochs_reward
        self.train_acc_threshold_reward = train_acc_threshold_reward
        self.batch_size_reward = batch_size_reward
        self.reward_model_kwargs = reward_model_kwargs
        self.lr_reward = learning_rate_reward
        self.n_steps_eval_current = n_steps_eval_current or self.schedule.n_steps_reward
        self.train_members_sequential = train_members_sequential
        self.validate_on_train = validate_on_train
        self.validate_on_current = validate_on_current
        self.validate_on_held_out = validate_on_held_out
        self.log_sampler_metrics = log_sampler_metrics
        self.ensemble_disjoint_data = ensemble_disjoint_data
        self.ensemble_agg_fn = ensemble_agg_fn
        self.num_samples_ep_age = num_samples_ep_age

        self.reward_model_cls = {
            'reward_model': RewardModel,
            'multi_head_reward_model': MultiHeadRewardModel,
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
            buffer=base_state.buffer,
            reward_model=self.reward_model
        )


    def _init_callback(self):
        super()._init_callback()

        self.ensemble_reward_buffer = [[] for _ in range(self.training_env.num_envs)]

        obs_size, act_size = self._get_input_sizes()
        self.reward_model = self.reward_model_cls(obs_size + act_size, **self.reward_model_kwargs).to(self.device)

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
        
        return {
            'loss': losses.mean().item(),
            'loss_correct': losses[correct].mean().item(),
            'loss_incorrect': losses[~correct].mean().item(),
            'accuracy': correct.mean(dtype=torch.float).item(),
            'real_loss': losses[real_idx].mean().item(),
            'real_acc': correct[real_idx].mean(dtype=torch.float).item(),
            'synth_loss': losses[synth_idx].mean().item(),
            'synth_acc': correct[synth_idx].mean(dtype=torch.float).item(),
        }


    def _train_module_epoch(self, module: nn.Module, dataset: Dataset, ep_ages_dataset: Optional[Dataset], module_is_ensemble: bool = False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')
        metrics = []

        for segments, preferences, weights, mask in dataloader:
            self.rew_optimizer.zero_grad()

            pred_rewards = module(segments)
            if module_is_ensemble:
                pred_rewards = self.ensemble_agg_fn(pred_rewards)

            losses, correct = self._compute_loss(pred_rewards, preferences, weights)
            losses.sum().backward()
            self.rew_optimizer.step()

            metrics.append(self._compute_batch_metrics(losses, correct, mask))

        if ep_ages_dataset is None:
            return metrics

        ages_dataloader = DataLoader(ep_ages_dataset, batch_size=self.batch_size_reward, shuffle=True, pin_memory=self.device.type == 'cuda')

        for segments, segment_ages in ages_dataloader:
            self.rew_optimizer.zero_grad()

            assert isinstance(module, MultiHeadMember)
            predictions = module.auxiliary(segments)

            criterion = nn.MSELoss()
            loss = criterion(predictions, segment_ages)
            loss.backward()
            self.rew_optimizer.step()

            metrics.append({'age_loss': loss})

        return metrics


    def _get_dataset(self):
        real_dataset = MaskedDataset(self.feed_buffer.get_dataset(), 0)
        synth_dataset = MaskedDataset(self.synth_buffer.get_dataset(), 1)
        return ConcatDataset([real_dataset, synth_dataset])


    def _get_episode_ages_dataset(self, num_samples: int):
        episodes = self.buffer.get_episodes()
        episode_ages = self.buffer.get_episode_ages()
        normalized_ep_ages = episode_ages / self.model._total_timesteps

        ep_indices, segments, _ = self.uniform_sampler.sample_segments(episodes, num_samples)
        segment_ages = normalized_ep_ages[ep_indices]
        return TensorDataset(segments, segment_ages)


    def _get_subset_for_member(self, dataset: Dataset, member_idx: int, length: int, ensemble_size: int):
        indices = range(member_idx, length, ensemble_size)
        return Subset(dataset, indices)


    def _train_reward_model_epoch(self):
        dataset = self._get_dataset()
        if self.num_samples_ep_age is not None:
            ep_ages_dataset = self._get_episode_ages_dataset(self.num_samples_ep_age)
        else:
            ep_ages_dataset = None

        if not self.train_members_sequential:
            metrics = self._train_module_epoch(self.reward_model, dataset, ep_ages_dataset, module_is_ensemble=True)
            return metrics

        metrics = []

        for index, member in enumerate(self.reward_model.members):
            if self.ensemble_disjoint_data:
                ensemble_size = len(self.reward_model.members)
                dataset_member = self._get_subset_for_member(dataset, index, len(dataset), ensemble_size)
                if ep_ages_dataset is not None:
                    ep_ages_dataset_member = self._get_subset_for_member(ep_ages_dataset, index, len(ep_ages_dataset), ensemble_size)
                member_metrics = self._train_module_epoch(member, dataset_member, ep_ages_dataset_member)
            else:
                member_metrics = self._train_module_epoch(member, dataset, ep_ages_dataset)

            metrics.append(member_metrics)

        return metrics


    def _train_predictor(self):
        self.reward_model.train()

        metrics = []
        for epoch in range(self.n_epochs_reward):
            epoch_metrics = self._train_reward_model_epoch()
            metrics.append(epoch_metrics)

            epoch_accuracy = np.mean([batch['accuracy'] for member in epoch_metrics for batch in member if 'accuracy' in batch])
            if epoch_accuracy > self.train_acc_threshold_reward:
                break

        grouped_metrics = defaultdict(list)
        for batch_metrics in [batch for epoch in metrics for member in epoch for batch in member]:
            for name, value in batch_metrics.items():
                grouped_metrics[name].append(value)

        avg_metrics = {key: np.mean(value) for key, value in grouped_metrics.items()}
        all_metrics = avg_metrics | {
            'epochs': epoch + 1,
        }
        log_metrics = {f"reward_model/train/{name}": value for name, value in all_metrics.items()}
        self.logger.record_with_progress(log_metrics, self.num_feed, self.training_progress)

        with torch.no_grad():
            if self.validate_on_train:
                self._validate_train()
            if self.validate_on_held_out:
                self._validate_held_out()


    def _calculate_calibration_metrics(self, pred_returns: torch.Tensor, preferences: torch.Tensor):
        inputs = pred_returns[..., 1] - pred_returns[..., 0]
        probs = torch.sigmoid(inputs.abs())
        
        pred_preferences = torch.argmax(pred_returns, dim=-1)
        correct_mask = (pred_preferences == preferences)
        incorrect_mask = ~correct_mask
        
        return {
            'confidence_correct': probs[correct_mask].mean().item() if correct_mask.any() else None,
            'confidence_incorrect': probs[incorrect_mask].mean().item() if incorrect_mask.any() else None,
        }


    def _compute_batch_metrics_validation(self, segments: torch.Tensor, preferences: torch.Tensor):
        segments = segments.to(self.device)
        preferences = preferences.to(self.device)
        
        ensemble_preds = self.reward_model(segments)
        
        member_metrics = []
        for member_pred in ensemble_preds:
            loss, correct = self._compute_loss(member_pred, preferences)
            member_returns = einops.reduce(member_pred, 'batch pair segment 1 -> batch pair', 'sum')
            calibration = self._calculate_calibration_metrics(member_returns, preferences)
            
            member_metrics.append({
                'loss': loss.mean().item(),
                'accuracy': correct.mean(dtype=torch.float).item(),
                **calibration
            })
        
        ensemble_pred = self.ensemble_agg_fn(ensemble_preds)
        loss, correct = self._compute_loss(ensemble_pred, preferences)
        ensemble_returns = einops.reduce(ensemble_pred, 'batch pair segment 1 -> batch pair', 'sum')
        calibration = self._calculate_calibration_metrics(ensemble_returns, preferences)

        return {
            'ensemble': {
                'loss': loss.mean().item(),
                'accuracy': correct.mean(dtype=torch.float).item(),
                **calibration
            },
            'members': member_metrics
        }


    def _aggregate_metrics(self, batch_metrics_list: list):
        ensemble_metrics = [b['ensemble'] for b in batch_metrics_list]
        member_metrics = [m for b in batch_metrics_list for m in b['members']]
        
        def avg_not_none(metrics_list, key):
            values = [m[key] for m in metrics_list if m.get(key) is not None]
            return np.mean(values) if values else 0
    
        metrics = {
            'loss': np.mean([m['loss'] for m in ensemble_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in ensemble_metrics]),
            'member_loss': np.mean([m['loss'] for m in member_metrics]),
            'member_accuracy': np.mean([m['accuracy'] for m in member_metrics]),
        }
        calibration_metrics = {
            'conf_good': avg_not_none(ensemble_metrics, 'confidence_correct'),
            'conf_bad': avg_not_none(ensemble_metrics, 'confidence_incorrect'),
            'member_conf_good': avg_not_none(member_metrics, 'confidence_correct'),
            'member_conf_bad': avg_not_none(member_metrics, 'confidence_incorrect'),
        }
        return {**metrics, **calibration_metrics}


    def _validate_on_segments(self, segments: torch.Tensor, rewards: torch.Tensor, preferences: torch.Tensor):
        self.reward_model.eval()
        
        dataset = TensorDataset(segments, preferences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size_reward, pin_memory=self.device.type == 'cuda')
        
        batch_metrics_list = []
        for batch_segments, batch_preferences in dataloader:
            batch_metrics = self._compute_batch_metrics_validation(batch_segments, batch_preferences)
            batch_metrics_list.append(batch_metrics)
        
        results = self._aggregate_metrics(batch_metrics_list)
        return results


    def _validate_on_episodes(self, episodes, episode_ages, size: int, compute_sampler_metrics: bool = True):
        schedule_state = self._create_schedule_state()
        segments, rewards, sampler_metrics = self.sampler.sample_pairs(episodes, episode_ages, size, reward_model=self.reward_model, compute_uniform_metrics=compute_sampler_metrics, schedule_state=schedule_state)
        preferences, keep_indices = self.eval_teacher.query_segments(rewards)
        return {
            **self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences),
            'sampler_metrics': sampler_metrics
        }


    def _log_validation_metrics(self, metrics: dict, prefix: str = ''):
        if self.log_sampler_metrics and metrics.get('sampler_metrics') is not None:
            self._log_metrics_stats(metrics.pop('sampler_metrics'), prefix=prefix)

        scoped_metrics = {key: value for key, value in metrics.items() if key != 'sampler_metrics' and value is not None}
        self.logger.record_with_progress(scoped_metrics, self.num_feed, self.training_progress, prefix=f'reward_model/eval/{prefix}')


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

        metrics = self._validate_on_segments(segments[keep_indices], rewards[:, keep_indices], preferences)
        self._log_validation_metrics({**metrics, 'sampler_metrics': sampler_metrics}, prefix='held_out/')

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
            return self.ensemble_agg_fn(ensemble_rewards)


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

                ensemble_metrics = {
                    'pref/ep_rew_mean': ep_info['pred_r_mean'],
                    'pref/step_rew_mean': ep_info['pred_r_step'],
                    'pref/ep_rew_std': ep_info['pred_r_std'],
                    'reward_model/avg_member_std_ep': ep_info['pred_r_std_member'],
                    'reward_model/avg_ensemble_std_rew': ep_info['pred_r_uncertainty'],
                }
                self.logger.record_with_progress(ensemble_metrics, self.num_feed, self.training_progress)


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
