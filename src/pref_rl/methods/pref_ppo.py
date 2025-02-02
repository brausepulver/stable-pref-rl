from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref import PrefCallback
from ..callbacks.unsuper import UnsuperCallback


class PrefPPO(PPO):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsuper = unsuper
        self.pref = pref


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [PrefCallback(**self.pref), UnsuperCallback(**self.unsuper)]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)

        return super().learn(*args, callback=callback_list, **kwargs)
