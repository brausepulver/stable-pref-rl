from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ..callbacks.direct import DIRECTCallback


class DIRECT(PPO):
    def __init__(self, *args, direct={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct = direct


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [DIRECTCallback(**self.direct)]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
