import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from ..callbacks.pref import PrefCallback
from ..callbacks.unsuper import UnsuperCallback


class PEBBLECallback(BaseCallback):
    def __init__(self, relabel_batch_size: int = 62_500, **kwargs):
        super().__init__(**kwargs)
        self.relabel_batch_size = relabel_batch_size


    def _on_step(self):
        replay_buffer = self.model.replay_buffer

        obs, act = torch.tensor(replay_buffer.observations, dtype=torch.float), torch.tensor(replay_buffer.actions, dtype=torch.float)
        state_actions = torch.cat([obs, act], dim=-1)

        with torch.no_grad():
            n_chunks = state_actions.shape[0] // self.relabel_batch_size
            pred_rewards = torch.cat([self.parent.reward_model(batch) for batch in state_actions.chunk(n_chunks)])
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
