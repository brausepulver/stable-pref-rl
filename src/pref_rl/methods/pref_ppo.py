from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref_ppo import PrefPPOCallback
from ..callbacks.unsupervised import UnsupervisedCallback


class PrefPPO(PPO):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, _init_setup_model=False, **kwargs)

        ep_completed = self.env.unwrapped.envs[0].spec.max_episode_steps
        rollout_completed = kwargs['n_steps']
        self.n_steps = ep_completed + rollout_completed
        self._setup_model()

        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.n_steps_default = kwargs['n_steps']


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
        self.n_steps = self.n_steps_default
        self._truncate_buffer(self.rollout_buffer, self.n_steps)
        return super().train()


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [PrefPPOCallback(**self.pref_kwargs), UnsupervisedCallback(**self.unsuper_kwargs)]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)

        return super().learn(*args, callback=callback_list, **kwargs)
