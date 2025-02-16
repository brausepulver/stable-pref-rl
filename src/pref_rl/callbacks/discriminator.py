from abc import ABC, abstractmethod
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .reward_mod import RewardModifierCallback
from ..utils.model import build_layered_module


class Discriminator(nn.Module):
    def __init__(self, input_dim, net_arch=[32, 32], activation_fn=nn.ReLU):
        super().__init__()
        self.layers = build_layered_module(input_dim, net_arch, activation_fn)

    def forward(self, x):
        return self.layers(x)


class BaseDiscriminatorCallback(RewardModifierCallback, ABC):
    def __init__(self,
        n_epochs_disc: int = 10,
        learning_rate_disc: float = 2e-4,
        batch_size_disc: int = 128,
        disc_kwargs: dict = {},
        reward_mixture_coef: float = 0.5,
        use_rewards_as_features: bool = True,
        log_prefix: str = 'discriminator/',
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)

        self.n_epochs_disc = n_epochs_disc
        self.batch_size_disc = batch_size_disc
        self.disc_kwargs = disc_kwargs
        self.lr_disc = learning_rate_disc
        self.reward_mixture_coef = reward_mixture_coef
        self.use_rewards_as_features = use_rewards_as_features


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()
        input_dim = obs_size + act_size + (1 if self.use_rewards_as_features else 0)

        self.discriminator = Discriminator(input_dim, **self.disc_kwargs)
        self.disc_loss = nn.BCEWithLogitsLoss()
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)


    @abstractmethod
    def _get_positive_samples(self):
        raise NotImplementedError


    @abstractmethod
    def _get_negative_samples(self, batch_size: int):
        raise NotImplementedError


    def _build_dataset(self):
        positive_samples = self._get_positive_samples()
        negative_samples = self._get_negative_samples(len(positive_samples))

        samples = torch.cat([positive_samples, negative_samples])
        labels = torch.cat([torch.ones(len(positive_samples)), torch.zeros(len(negative_samples))])
        return TensorDataset(samples, labels)


    def _compute_disc_loss(self, inputs, labels):
        logits = self.discriminator(inputs).squeeze()
        loss = self.disc_loss(logits, labels)

        pred_labels = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (pred_labels == labels).float().mean().item()

        return loss, accuracy


    def _train_discriminator(self):
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


    def _predict_rewards(self):
        self.discriminator.eval()

        obs, act, gt_reward = self._get_current_step()
        disc_features = torch.cat([obs, act] + ([gt_reward] if self.use_rewards_as_features else []), dim=-1)

        with torch.no_grad():
            disc_reward = self.discriminator(disc_features)
            mixed_reward = self.reward_mixture_coef * disc_reward + (1 - self.reward_mixture_coef) * gt_reward

        return mixed_reward
