import itertools

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class LogRolloutStatsCallback(BaseCallback):
    def __init__(self, ep_info_log_keys: dict[str, str], window_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.ep_info_log_keys = ep_info_log_keys
        self.window_size = window_size

    def _init_callback(self):
        self.window_size = self.window_size or self.training_env.num_envs

    def _on_rollout_start(self) -> None:
        self.rollout_done_eps_count = 0

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0 or len(self.model.ep_info_buffer[0]) == 0:
            return

        if self.rollout_done_eps_count == 0:
            return

        ep_infos = list(itertools.islice(reversed(self.model.ep_info_buffer), self.window_size))
        for ep_info_key, log_key in self.ep_info_log_keys.items():
            values = [ep_info[ep_info_key] for ep_info in ep_infos if ep_info_key in ep_info]
            if len(values) > 0:
                self.logger.record(log_key, safe_mean(values))

    def _on_step(self):
        self.rollout_done_eps_count += self.locals['dones'].sum().item()
        return True


def get_default_callbacks():
    ep_info_log_keys = {
        'intr_r': 'pretrain/ep_intr_rew_mean',
        'pred_r_mean': 'pref/ep_pred_rew_mean',
        'pred_r_step': 'pref/step_pred_rew_mean',
        'pred_r_std': 'pref/ep_pred_rew_std',
        'pred_r_std_member': 'reward_model/avg_member_std_ep',
        'pred_r_uncertainty': 'reward_model/avg_ensemble_std_step',
        'r': 'rollout/ep_rew_mean_current',
        'pred_age': 'pref/pred_age',
    }
    return LogRolloutStatsCallback(ep_info_log_keys)
