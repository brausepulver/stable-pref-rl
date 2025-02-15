from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.direct import DIRECTCallback
from ..utils.callbacks import get_default_callbacks


class DIRECT(PPO):
    def __init__(self, *args, direct={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_kwargs = direct


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [DIRECTCallback(**self.direct_kwargs), get_default_callbacks()]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
