import itertools

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class LogRolloutStatsCallback(BaseCallback):
    def __init__(self, ep_info_log_keys: dict[str, str], window_size: int = None, suffix_window_size: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ep_info_log_keys = ep_info_log_keys
        self.window_size = window_size
        self.suffix_window_size = f"{window_size}_eps" if suffix_window_size else ""

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
        for ep_info_key, (log_prefix, log_key) in self.ep_info_log_keys.items():
            self._log_data_for_key(ep_infos, ep_info_key, log_prefix, log_key)

    def _log_data_for_key(self, ep_infos: list, ep_info_key: str, log_prefix: str, log_key: str):
        items = [ep_info[ep_info_key] for ep_info in ep_infos if ep_info_key in ep_info]
        if len(items) > 0:
            self.logger.record(f"{log_prefix}/{log_key}{self.suffix_window_size}", safe_mean(items))

    def _on_step(self):
        self.rollout_done_eps_count += self.locals['dones'].sum().item()
        return True


def get_default_callbacks():
    ep_info_log_keys = {
        'intr_r': ('pretrain', 'ep_intr_rew_mean'),
        'pred_r_mean': ('pref', 'ep_rew_mean'),
        'pred_r_step': ('pref', 'step_rew_mean'),
        'pred_r_std': ('pref', 'ep_rew_std'),
        'pred_r_std_member': ('reward_model', 'avg_member_std_ep'),
        # 'pred_r_coef_var': ('reward_model', 'pred_rew_coef_var'),
        'pred_r_uncertainty': ('reward_model', 'avg_ensemble_std_step'),
        # 'pred_r_uncertainty_coef_var': ('reward_model/', 'rew_model_coef_var'),
        'r': ('rollout', 'ep_rew_mean_current')
    }
    return LogRolloutStatsCallback(ep_info_log_keys)
