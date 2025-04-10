import einops
import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .reward_mod import RewardModifierCallback
from ..utils.pref import EpisodeBuffer, Teacher, Sampler
from ..utils.model import build_layered_module


class Discriminator(nn.Module):
    def __init__(self, input_dim, net_arch=[32, 32], activation_fn=nn.ReLU()):
        super().__init__()
        self.layers = build_layered_module(input_dim, net_arch, activation_fn)

    def forward(self, x):
        return self.layers(x)


class PrefDIRECTCallback(RewardModifierCallback):
    def __init__(self,
        n_steps_reward: int = 32_000,
        ann_buffer_size_eps: int = None,
        sampler: str = 'uniform',
        segment_size: int = 50,
        max_feed: int = 2_000,
        feed_batch_size: int = 200,
        pref_buffer_size_seg: int = 400,
        teacher: str = None,
        teacher_kwargs: dict = {},
        n_steps_first_train: int = None,
        on_first_trained: callable = None,
        on_trained: callable = None,
        margins_stats_window_size: int = 100,
        n_epochs_disc: int = 10,
        learning_rate_disc: float = 3e-4,
        batch_size_disc: int = 128,
        disc_kwargs: dict = {},
        reward_mixture_coef: float = 0.5,
        use_rewards_as_features: bool = False,
        log_prefix: str = 'discriminator/',
        **kwargs
    ):
        super().__init__(log_prefix=log_prefix, **kwargs)

        self.n_steps_reward = n_steps_reward
        self.ann_buffer_size_eps = ann_buffer_size_eps
        self.sampling_method = sampler
        self.segment_size = segment_size
        self.max_feed = max_feed
        self.feed_batch_size = feed_batch_size
        self.pref_buffer_size_seg = pref_buffer_size_seg
        self.train_teacher = teacher
        self.teacher_kwargs = teacher_kwargs
        self.n_steps_first_train = n_steps_first_train
        self.on_first_trained = on_first_trained
        self.on_trained = on_trained
        self.margins_stats_window_size = margins_stats_window_size

        self.n_epochs_disc = n_epochs_disc
        self.batch_size_disc = batch_size_disc
        self.disc_kwargs = disc_kwargs
        self.lr_disc = learning_rate_disc
        self.reward_mixture_coef = reward_mixture_coef
        self.use_rewards_as_features = use_rewards_as_features

        assert self.ann_buffer_size_eps is None or self.margins_stats_window_size <= self.ann_buffer_size_eps

        self.has_trained = False
        self.num_feed = 0


    def _init_callback(self):
        super()._init_callback()

        obs_size, act_size = self._get_input_sizes()

        self.buffer = EpisodeBuffer(self.training_env.num_envs, self.ann_buffer_size_eps)
        self.sampler = Sampler(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size)
        self.train_teacher = Teacher(segment_size=self.segment_size, observation_size=obs_size, action_size=act_size, teacher=self.train_teacher, teacher_kwargs=self.teacher_kwargs)

        segment_dim = obs_size + act_size
        self.segment_buffer = torch.empty((0, 2, self.segment_size, segment_dim))
        self.preference_buffer = torch.empty((0,))

        input_dim = obs_size + act_size + (1 if self.use_rewards_as_features else 0)
        self.discriminator = Discriminator(input_dim, **self.disc_kwargs)
        self.disc_loss = nn.BCEWithLogitsLoss()
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)


    def _get_predictor(self):
        return self.discriminator


    def _get_steps_from_preferences(self, preferences: torch.Tensor):
        indices = (torch.arange(len(self.segment_buffer)), preferences.to(dtype=torch.long))
        segments = self.segment_buffer[indices]
        steps = einops.rearrange(segments, 'l s d -> (l s) d')
        return steps


    def _get_positive_samples(self, size: int = None):
        steps = self._get_steps_from_preferences(self.preference_buffer == 1)
        indices = torch.randperm(len(steps))[:size]
        return steps[indices]


    def _get_negative_samples(self, size: int = None):
        obs_size, act_size = self._get_input_sizes()
        recent_steps = torch.cat(list(self.buffer.episodes))[-size:]
        steps, _ = torch.split(recent_steps, (obs_size + act_size, 1), dim=-1)
        return steps


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


    def _expand_data(self, sampling_method: str):
        num_samples = min(self.feed_batch_size, self.max_feed - self.num_feed)
        state_actions, rewards = self.sampler.sample_segments(self.buffer.episodes, num_samples, sampling_method, self._get_predictor())

        preferences, keep_indices = self.train_teacher.query_segments(rewards)

        self.segment_buffer = torch.cat([self.segment_buffer, state_actions[keep_indices]])[-self.pref_buffer_size_seg:]
        self.preference_buffer = torch.cat([self.preference_buffer, preferences])[-self.pref_buffer_size_seg:]

        self.num_feed += len(keep_indices)
        self.logger.record('pref/num_feed', self.num_feed)


    def _predict_rewards(self):
        self.discriminator.eval()

        obs, act, gt_reward = self._get_current_step()
        disc_features = torch.cat([obs, act] + ([gt_reward] if self.use_rewards_as_features else []), dim=-1)

        with torch.no_grad():
            disc_reward = self.discriminator(disc_features)
            mixed_reward = self.reward_mixture_coef * disc_reward + (1 - self.reward_mixture_coef) * gt_reward

        return mixed_reward


    def _on_step(self):
        obs, act, gt_rewards = self._get_current_step()
        annotations = torch.cat([obs, act, gt_rewards], dim=-1)
        self.buffer.add(annotations, self.locals['dones'])

        if self.locals['dones'].any():
            recent_eps = list(itertools.islice(reversed(self.buffer.episodes), self.margins_stats_window_size))
            self.train_teacher.update_thresholds(recent_eps)

        buffer_empty = len(self.buffer.episodes) == 0
        if not buffer_empty and self.n_steps_first_train is None:
            self.n_steps_first_train = self.num_timesteps

        should_train = self.n_steps_first_train is not None
        checkpoint_reached = should_train and (self.num_timesteps - self.n_steps_first_train) % self.n_steps_reward == 0
        feedback_left = self.num_feed < self.max_feed

        if checkpoint_reached:
            sampling_method = 'uniform' if not self.has_trained else self.sampling_method
            if feedback_left:
                self._expand_data(sampling_method)
            self._train_predictor()

            if self.on_trained:
                self.on_trained()

            if not self.has_trained:
                self.has_trained = True
                if self.on_first_trained: self.on_first_trained()

            self.logger.dump(step=self.num_timesteps)

        return True if not self.has_trained else super()._on_step()
