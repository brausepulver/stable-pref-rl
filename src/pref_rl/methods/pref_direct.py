from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.pref_direct import PrefDIRECTCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks


class PrefDIRECT(PPO):
    def __init__(self, *args, unsuper={}, pref={}, direct={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.direct_kwargs = direct


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [
            PrefDIRECTCallback(pref_kwargs=self.pref_kwargs, direct_kwargs=self.direct_kwargs),
            UnsupervisedCallback(**self.unsuper_kwargs),
            get_default_callbacks()
        ]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
