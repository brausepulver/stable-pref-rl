import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples


from ..utils.logger import create_pref_logger
from ..callbacks.pref_ppo import PrefPPOCallback
from ..utils.callbacks import get_default_callbacks
from ..utils.entropy import compute_neighbor_distances


class EntropyReplayBuffer(ReplayBuffer):
    """Replay buffer that can replace sampled rewards with state-entropy targets."""

    def __init__(self, *args, n_neighbors: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.chunk_size = 1024
        self.entropy_active = False


    def set_entropy_params(self, n_neighbors: int | None = None, chunk_size: int | None = None):
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if chunk_size is not None:
            self.chunk_size = chunk_size


    def set_entropy_active(self, active: bool):
        self.entropy_active = active


    def _compute_entropy_rewards(self, observations: torch.Tensor) -> torch.Tensor:
        buffer_len = self.buffer_size if self.full else self.pos
        reference = torch.as_tensor(self.observations[:buffer_len], device=observations.device, dtype=observations.dtype)
        reference = reference.reshape(buffer_len * self.n_envs, -1)

        neighbor_dist = compute_neighbor_distances(
            observations,
            reference,
            n_neighbors=self.n_neighbors,
            chunk_size=self.chunk_size,
        )

        std = neighbor_dist.std()
        if std.item() == 0:
            std = torch.tensor(1.0, device=neighbor_dist.device)

        rewards = neighbor_dist / std
        return rewards.unsqueeze(-1)


    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        data = super().sample(batch_size, env)

        if not self.entropy_active:
            return data

        with torch.no_grad():
            rewards = self._compute_entropy_rewards(data.observations)

        return ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=rewards.to(self.device),
        )


class PEBBLE(SAC):

    def __init__(self, *args, n_steps_seed: int = 1000, reset_gradient_steps: int = 100, unsuper={}, pref={}, **kwargs):
        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref

        replay_buffer_kwargs = {}
        if 'n_neighbors' in self.unsuper_kwargs:
            replay_buffer_kwargs['n_neighbors'] = self.unsuper_kwargs['n_neighbors']

        kwargs['replay_buffer_class'] = EntropyReplayBuffer

        super().__init__(*args, **kwargs)

        self.n_steps_seed = n_steps_seed
        self.n_steps_unsuper = self.unsuper_kwargs.get('n_steps_unsuper', None)
        self.unsuper_enabled = self.n_steps_unsuper is not None
        self.reset_gradient_steps = reset_gradient_steps
        self.relabel_batch_size = 64

        self.replay_buffer.set_entropy_active(False)


    def _reset_critics(self):
        """Re-initialize the critic networks while preserving the actor state."""
        critic_features_extractor = self.policy.actor.features_extractor if self.policy.share_features_extractor else None

        if self.policy.share_features_extractor:
            critic = self.policy.make_critic(features_extractor=critic_features_extractor)
            critic_parameters = [
                param
                for name, param in critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            critic = self.policy.make_critic(features_extractor=None)
            critic_parameters = list(critic.parameters())

        critic_target = self.policy.make_critic(features_extractor=None)
        critic_target.load_state_dict(critic.state_dict())

        critic.optimizer = self.policy.optimizer_class(
            critic_parameters,
            lr=self.lr_schedule(1),
            **self.policy.optimizer_kwargs,
        )
        critic_target.set_training_mode(False)

        self.policy.critic = critic
        self.policy.critic_target = critic_target

        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])


    def _setup_logger(self, reset_num_timesteps: bool = True, tb_log_name: str = "run", wandb_run = None):
        logger = create_pref_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps, wandb_run)
        self.set_logger(logger)


    def _setup_learn(self, total_timesteps, callback, reset_num_timesteps=True, tb_log_name="run", progress_bar=False):
        self._setup_logger(reset_num_timesteps, tb_log_name)
        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)


    def train(self, *args, **kwargs):
        if self.unsuper_enabled and (self.n_steps_seed <= self.num_timesteps < self.n_steps_seed + self.n_steps_unsuper):
            self.replay_buffer.set_entropy_active(True)

        if self.unsuper_enabled and self.num_timesteps >= self.n_steps_seed + self.n_steps_unsuper:
            self.replay_buffer.set_entropy_active(False)

        return super().train(*args, **kwargs)


    def learn(self, *args, callback=None, **kwargs):
        def on_first_trained():
            """Reset the critics and run an initial round of updates once preferences begin."""
            self._reset_critics()
            if self.reset_gradient_steps:
                self.train(gradient_steps=self.reset_gradient_steps, batch_size=self.batch_size)

        def relabel_replay_buffer():
            buffer_pos = self.replay_buffer.observations.shape[0] if self.replay_buffer.full else self.replay_buffer.pos

            obs = torch.tensor(self.replay_buffer.observations[:buffer_pos], dtype=torch.float)
            act = torch.tensor(self.replay_buffer.actions[:buffer_pos], dtype=torch.float)
            state_actions = torch.cat([obs, act], dim=-1)

            with torch.no_grad():
                n_chunks = max(1, state_actions.shape[0] // self.relabel_batch_size)
                pred_rewards = torch.cat([
                    self.pref_ppo_callback.reward_model(batch.to(self.pref_ppo_callback.device)).detach()
                    for batch in state_actions.chunk(n_chunks)
                ], dim=1).squeeze(-1)
                self.replay_buffer.rewards[:buffer_pos] = pred_rewards.mean(dim=0).cpu().numpy()

        self.pref_ppo_callback = PrefPPOCallback(
            on_first_trained=on_first_trained,
            on_trained=relabel_replay_buffer,
            **self.pref_kwargs
        )

        self.pref_ppo_callback.schedule.n_steps_first_train = self.n_steps_seed + self.n_steps_unsuper

        callbacks = [self.pref_ppo_callback, get_default_callbacks()]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)

        return super().learn(*args, callback=callback_list, **kwargs)
