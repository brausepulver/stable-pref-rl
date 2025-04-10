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
        def on_first_trained():
            self.policy.init_weights(self.policy.value_net)

        callbacks = [
            PrefDIRECTCallback(
                n_steps_first_train=self.unsuper_kwargs['n_steps_unsuper'],
                on_first_trained=on_first_trained,
                **self.pref_kwargs,
                **self.direct_kwargs
            ),
            UnsupervisedCallback(**self.unsuper_kwargs),
            get_default_callbacks()
        ]

        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)
