import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_parameters_by_name
from ..callbacks.pref_ppo import PrefPPOCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks


class PEBBLE(SAC):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.relabel_batch_size = 64


    def learn(self, *args, callback=None, **kwargs):
        def on_first_trained():
            """
            Reset critic networks, log entropy coefficient and optimizers after unsupervised pretraining is complete.
            This code is largely based on the SAC._setup_model() method, with the exception that we save and restore the actor weights.
            """
            trained_actor = self.policy.actor
            self.policy._build(self.lr_schedule)
            self.policy.actor = trained_actor

            self._create_aliases()

            self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
            self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

            if self.target_entropy == "auto":
                self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
            else:
                self.target_entropy = float(self.target_entropy)

            if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
                init_value = 1.0
                if "_" in self.ent_coef:
                    init_value = float(self.ent_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
                self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
            else:
                self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

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
            n_steps_first_train=self.unsuper_kwargs['n_steps_unsuper'],
            on_first_trained=on_first_trained,
            on_trained=relabel_replay_buffer,
            **self.pref_kwargs
        )

        callbacks = [
            self.pref_ppo_callback,
            UnsupervisedCallback(**self.unsuper_kwargs),
            get_default_callbacks()
        ]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
