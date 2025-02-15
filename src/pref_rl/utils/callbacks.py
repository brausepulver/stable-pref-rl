from .logging import LogRolloutStatsCallback


def get_default_callbacks():
    ep_info_log_keys = {'intr_r': 'ep_intr_rew_mean', 'pred_r': 'ep_pred_rew_mean', 'r': 'ep_rew_mean'}
    return LogRolloutStatsCallback(ep_info_log_keys)
