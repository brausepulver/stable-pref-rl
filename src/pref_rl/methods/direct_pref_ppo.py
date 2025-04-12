from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref_direct import PrefDIRECTCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks


class DIRECTPrefPPO(PPO):
    def __init__(self, *args, unsuper={}, pref={}, direct={}, **kwargs):
        super().__init__(*args, _init_setup_model=False, **kwargs)

        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.direct_kwargs = direct
        self.unsuper_enabled = self.unsuper_kwargs['n_steps_unsuper'] is not None

        if not self.unsuper_enabled:
            self.n_steps_default = kwargs['n_steps']
            ep_completed = self.env.unwrapped.envs[0].spec.max_episode_steps
            rollout_completed = self.n_steps_default
            self.n_steps = ep_completed + rollout_completed

        self._setup_model()


    def _truncate_buffer(self, buffer, size: int):
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
        def on_first_trained():
            self.policy.init_weights(self.policy.value_net)

        callbacks = [
            PrefDIRECTCallback(
                n_steps_first_train=self.unsuper_kwargs['n_steps_unsuper'] or None,
                on_first_trained=on_first_trained,
                **self.pref_kwargs,
                **self.direct_kwargs
            ),
            *([UnsupervisedCallback(**self.unsuper_kwargs)] if self.unsuper_enabled else []),
            get_default_callbacks()
        ]

        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
