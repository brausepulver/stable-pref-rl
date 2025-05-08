import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref_ppo import PrefPPOCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks


class PrefPPO(PPO):
    def __init__(self, *args, save_final_pref_buffer=False, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, _init_setup_model=False, **kwargs)

        self.save_final_pref_buffer = save_final_pref_buffer
        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.unsuper_enabled = self.unsuper_kwargs['n_steps_unsuper'] is not None

        if not self.unsuper_enabled:
            self.n_steps_default = kwargs['n_steps']
            ep_completed = self.env.unwrapped.envs[0].spec.max_episode_steps
            rollout_completed = self.n_steps_default
            self.n_steps = ep_completed + rollout_completed

        self._setup_model()

        def on_first_trained():
            self.policy.init_weights(self.policy.value_net)

        self.pref_ppo_callback = PrefPPOCallback(
            n_steps_first_train=self.unsuper_kwargs['n_steps_unsuper'] or None,
            on_first_trained=on_first_trained,
            **self.pref_kwargs
        )
        self.callbacks = [
            self.pref_ppo_callback,
            *([UnsupervisedCallback(**self.unsuper_kwargs)] if self.unsuper_enabled else []),
            get_default_callbacks()
        ]


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
        callback_list = CallbackList(([callback] if callback is not None else []) + self.callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)


    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['callbacks']


    def _get_torch_save_params(self):
        ppo_state_dicts, ppo_vars = super()._get_torch_save_params()
        state_dicts = ['pref_ppo_callback.reward_model', 'pref_ppo_callback.rew_optimizer']
        return ppo_state_dicts + state_dicts, ppo_vars


    def save(self, path, exclude = None, include = None):
        if self.save_final_pref_buffer:
            with open('done_eps.pkl', 'wb') as f:
                pickle.dump(self.pref_ppo_callback.buffer.done_eps, f)

        return super().save(path, exclude, include)
