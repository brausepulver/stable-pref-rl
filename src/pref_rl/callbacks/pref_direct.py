import einops
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .pref import BasePrefCallback
from ..utils.model import build_layered_module


class Discriminator(nn.Module):
    def __init__(self, input_dim, net_arch=[256, 256, 256], activation_fn=nn.ReLU()):
        super().__init__()
        self.layers = build_layered_module(input_dim, net_arch, activation_fn)

    def forward(self, x):
        return self.layers(x)


class PrefDIRECTCallback(BasePrefCallback):
    def __init__(self,
        n_epochs_disc: int = 10,
        learning_rate_disc: float = 3e-4,
        batch_size_disc: int = 128,
        disc_kwargs: dict = {},
        reward_mixture_coef: float = 1.0,
        use_rewards_as_features: bool = False,
        pref_buffer_size_seg: int = 200,
        keep_training: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs, log_prefix='direct/')

        self.n_epochs_disc = n_epochs_disc
        self.batch_size_disc = batch_size_disc
        self.disc_kwargs = disc_kwargs
        self.lr_disc = learning_rate_disc
        self.reward_mixture_coef = reward_mixture_coef
        self.use_rewards_as_features = use_rewards_as_features
        self.pref_buffer_size_seg = pref_buffer_size_seg
        self.keep_training = keep_training


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()
        input_dim = obs_size + act_size + (1 if self.use_rewards_as_features else 0)
        self.discriminator = Discriminator(input_dim, **self.disc_kwargs).to(self.device)
        self.disc_loss = nn.BCEWithLogitsLoss()
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)


    def _get_predictor(self):
        return self.discriminator


    def _expand_data(self, sampling_method: str):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.get_episodes(), num_samples, sampling_method, self._get_predictor())

        preferences, keep_indices = self.train_teacher.query_segments(rewards.detach())

        self.segment_buffer = torch.cat([self.segment_buffer, state_actions[keep_indices].detach().to(self.device)])[-self.pref_buffer_size_seg:]
        self.preference_buffer = torch.cat([self.preference_buffer, preferences.detach().to(self.device)])[-self.pref_buffer_size_seg:]

        self.num_feed += len(keep_indices)
        self.logger.record('pref/num_feed', self.num_feed)


    def _get_steps_from_preferences(self, preferences: torch.Tensor):
        indices = (torch.arange(len(self.segment_buffer), device=self.device),
                  preferences.to(dtype=torch.long))
        segments = self.segment_buffer[indices]
        steps = einops.rearrange(segments, 'l s d -> (l s) d')
        return steps


    def _get_positive_samples(self, size: int = None):
        steps = self._get_steps_from_preferences(self.preference_buffer == 1)
        indices = torch.randperm(len(steps), device=self.device)[:size]
        return steps[indices]


    def _get_negative_samples(self, size: int = None):
        obs_size, act_size = self._get_input_sizes()
        recent_steps = torch.cat(list(self.buffer.get_episodes()))[-size:].to(self.device)
        steps, _ = torch.split(recent_steps, (obs_size + act_size, 1), dim=-1)
        return steps


    def _build_dataset(self):
        positive_samples = self._get_positive_samples()
        negative_samples = self._get_negative_samples(len(positive_samples))
        samples = torch.cat([positive_samples, negative_samples])
        labels = torch.cat([
            torch.ones(len(positive_samples), device=self.device),
            torch.zeros(len(negative_samples), device=self.device)
        ])
        return TensorDataset(samples, labels)


    def _compute_disc_loss(self, inputs, labels):
        logits = self.discriminator(inputs).squeeze()
        loss = self.disc_loss(logits, labels)

        pred_labels = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (pred_labels == labels).float().mean().item()

        return loss, accuracy


    def _train_predictor(self):
        self.discriminator.train()

        dataloader = DataLoader(self._build_dataset(), batch_size=self.batch_size_disc, shuffle=True)
        losses = []
        accuracies = []

        for _ in range(self.n_epochs_disc):
            for inputs, labels in dataloader:
                self.disc_optimizer.zero_grad()
                loss, accuracy = self._compute_disc_loss(inputs, labels)
                loss.backward()
                self.disc_optimizer.step()

                losses.append(loss.item())
                accuracies.append(accuracy)

        self.logger.record('discriminator/train/loss', np.mean(losses))
        self.logger.record('discriminator/train/accuracy', np.mean(accuracies))
        self.logger.record('discriminator/train/epochs', self.n_epochs_disc)


    def _predict_rewards(self):
        self.discriminator.eval()

        obs, act, gt_reward = self._get_current_step()
        disc_features = torch.cat([obs, act] + ([gt_reward] if self.use_rewards_as_features else []), dim=-1).to(self.device)

        with torch.no_grad():
            disc_reward = self.discriminator(disc_features)
            mixed_reward = self.reward_mixture_coef * disc_reward + (1 - self.reward_mixture_coef) * gt_reward.to(self.device)

        return mixed_reward.detach()
