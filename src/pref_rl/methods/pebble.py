import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from ..callbacks.pref import PrefCallback
from ..callbacks.unsuper import UnsuperCallback


class PEBBLECallback(BaseCallback):
    def _on_step(self):
        replay_buffer = self.model.replay_buffer

        obs, act = torch.tensor(replay_buffer.observations, dtype=torch.float), torch.tensor(replay_buffer.observations, dtype=torch.float)
        state_actions = torch.cat([obs, act], dim=-1)

        pred_rewards = self.parent.reward_model(state_actions)
        self.model.replay_buffer.rewards = pred_rewards.numpy()


class PEBBLE(SAC):
    def __init__(self, *args, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsuper = unsuper
        self.pref = pref


    def learn(self, *args, callback=None, **kwargs):
        callbacks = [PrefCallback(**self.pref, callback=PEBBLECallback()), UnsuperCallback(**self.unsuper)]
        callback_list = CallbackList(([callback] if callback is not None else []) + callbacks)

        return super().learn(*args, callback=callback_list, **kwargs)
