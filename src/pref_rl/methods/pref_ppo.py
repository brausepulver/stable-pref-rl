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
        pref_callback = PrefCallback(**self.pref)
        unsuper_callback = UnsuperCallback(callback=pref_callback, **self.unsuper)

        callback = ([callback] if callback is not None else []) + [unsuper_callback]
        callback = CallbackList(callback)

        return super().learn(*args, callback=callback, **kwargs)
