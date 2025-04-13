from .logging import LogRolloutStatsCallback


def get_default_callbacks():
    ep_info_log_keys = {'intr_r': ('pretrain', 'ep_intr_rew_mean'), 'pred_r': ('pref', 'ep_pred_rew_mean'), 'r': ('rollout', 'ep_rew_mean')}
    return LogRolloutStatsCallback(ep_info_log_keys)
