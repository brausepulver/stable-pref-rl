from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref_ppo import PrefPPOCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks


class PrefPPO(PPO):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, _init_setup_model=False, **kwargs)

        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.unsuper_enabled = self.unsuper_kwargs['n_steps_unsuper'] > 0

        if not self.unsuper_enabled:
            self.n_steps_default = kwargs['n_steps']
            ep_completed = self.env.unwrapped.envs[0].spec.max_episode_steps
            rollout_completed = self.n_steps_default
            self.n_steps = ep_completed + rollout_completed

        self._setup_model()


    def _truncate_buffer(self, buffer: RolloutBuffer, size: int):
        buffer.full = True
        buffer.buffer_size = size

        buffer.observations = buffer.observations[-size:]
        buffer.actions = buffer.actions[-size:]
        buffer.values = buffer.values[-size:]
        buffer.log_probs = buffer.log_probs[-size:]
        buffer.advantages = buffer.advantages[-size:]
        buffer.returns = buffer.returns[-size:]


    def train(self):
        if not self.unsuper_enabled:
            self.n_steps = self.n_steps_default
            self._truncate_buffer(self.rollout_buffer, self.n_steps)

        return super().train()


    def learn(self, *args, callback=None, **kwargs):
        def on_first_pref_ppo_train():
            self.policy.init_weights(self.policy.value_net)

        callbacks = [
            PrefPPOCallback(
                n_steps_first_train=self.unsuper_kwargs['n_steps_unsuper'],
                on_first_train=on_first_pref_ppo_train,
                **self.pref_kwargs
            ),
            UnsupervisedCallback(**self.unsuper_kwargs),
            get_default_callbacks()
        ]

        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
