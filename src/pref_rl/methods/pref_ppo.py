from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList

from ..callbacks.pref_ppo import PrefPPOCallback
from ..callbacks.unsupervised import UnsupervisedCallback
from ..utils.callbacks import get_default_callbacks
from ..utils.logger import create_pref_logger
from ..policies.shared import SharedMlpActorCriticPolicy


class PrefPPO(PPO):

    policy_aliases = {
        **PPO.policy_aliases,
        "SharedMlpActorCriticPolicy": SharedMlpActorCriticPolicy
    }


    def __init__(self, *args, run_id: str | None = None, save_callback_data=False, save_episode_data=False, unsuper={}, pref={}, **kwargs):
        super().__init__(*args, _init_setup_model=False, **kwargs)

        self.run_id = run_id
        self.save_callback_data = save_callback_data
        self.save_episode_data = save_episode_data
        self.unsuper_kwargs = unsuper
        self.pref_kwargs = pref
        self.unsuper_enabled = self.unsuper_kwargs['n_steps_unsuper'] is not None

        if not self.unsuper_enabled:
            self.n_steps_default = kwargs['n_steps']
            ep_completed = self.env.unwrapped.envs[0].spec.max_episode_steps
            rollout_completed = self.n_steps_default
            self.n_steps = ep_completed + rollout_completed

        self._setup_model()
        self._setup_callbacks()


    def _setup_callbacks(self):
        def on_first_trained():
            self.policy.init_weights(self.policy.value_net)

        self.pref_ppo_callback = PrefPPOCallback(
            on_first_trained=on_first_trained,
            save_episode_data=self.save_episode_data,
            **self.pref_kwargs
        )
        if self.unsuper_enabled:
            self.pref_ppo_callback.schedule.n_steps_first_train = self.unsuper_kwargs['n_steps_unsuper']

        self.unsupervised_callback = UnsupervisedCallback(**self.unsuper_kwargs) if self.unsuper_enabled else None

        self.callbacks = [
            self.pref_ppo_callback,
            *([self.unsupervised_callback] if self.unsupervised_callback is not None else []),
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


    def _setup_logger(self, reset_num_timesteps: bool = True, tb_log_name: str = "run", wandb_run = None):
        logger = create_pref_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps, wandb_run)
        self.set_logger(logger)


    def _setup_learn(self, total_timesteps, callback, reset_num_timesteps=True, tb_log_name="run", progress_bar=False):
        self._setup_logger(reset_num_timesteps, tb_log_name)
        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)


    def learn(self, *args, callback=None, **kwargs):
        callback_list = CallbackList(([callback] if callback is not None else []) + self.callbacks)
        return super().learn(*args, callback=callback_list, **kwargs)


    def _excluded_save_params(self):
        excluded = super()._excluded_save_params()
        excluded.remove('rollout_buffer')
        excluded.extend(['pref_ppo_callback', 'unsupervised_callback', 'callbacks'])
        return excluded


    def _get_torch_save_params(self):
        ppo_state_dicts, ppo_vars = super()._get_torch_save_params()
        state_dicts = ['pref_ppo_callback.reward_model', 'pref_ppo_callback.rew_optimizers'] if self.save_callback_data else []
        vars = ['pref_ppo_callback.feed_buffer', 'pref_ppo_callback.synth_buffer'] if self.save_callback_data else []
        return ppo_state_dicts + state_dicts, ppo_vars + vars


    def save(self, *args, **kwargs):
        if self.save_episode_data:
            self.pref_ppo_data = {
                'buffer': self.pref_ppo_callback.buffer,
            }

        if self.save_callback_data:
            self.pref_ppo_data = {
                'ensemble_reward_buffer': self.pref_ppo_callback.ensemble_reward_buffer,
                'steps_since_train': self.pref_ppo_callback.steps_since_train,
                'has_trained': self.pref_ppo_callback.has_trained,
                'training_progress': self.pref_ppo_callback.training_progress,
                'n_steps_train_total': self.pref_ppo_callback.n_steps_train_end,
                'buffer': self.pref_ppo_callback.buffer,
                'train_teacher': self.pref_ppo_callback.train_teacher,
            }
            self.unsupervised_data = {
                '_buffer': self.unsupervised_callback._buffer,
            }

        return super().save(*args, **kwargs)


    @classmethod
    def load(cls, *args, **kwargs) -> "PrefPPO":
        model: PrefPPO = super().load(*args, **kwargs)

        if hasattr(model, "pref_ppo_data"):
            d, cb = model.pref_ppo_data, model.pref_ppo_callback
            cb.ensemble_reward_buffer = d["ensemble_reward_buffer"]
            cb.steps_since_train      = d["steps_since_train"]
            cb.has_trained            = d["has_trained"]
            cb.training_progress      = d["training_progress"]
            cb.n_steps_train_total    = d["n_steps_train_end"]
            cb.buffer                 = d["buffer"]
            cb.train_teacher          = d["train_teacher"]

        if hasattr(model, "unsupervised_data") and model.unsupervised_callback is not None:
            model.unsupervised_callback._buffer = model.unsupervised_data["_buffer"]

        return model
