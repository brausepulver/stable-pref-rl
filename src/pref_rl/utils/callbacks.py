from .logging import LogRolloutStatsCallback


def get_default_callbacks():
    ep_info_log_keys = {
        'intr_r': ('pretrain', 'ep_intr_rew_mean'),
        'pred_r': ('pref', 'ep_pred_rew_mean'),
        'pred_r_step': ('pref', 'pred_rew_mean'),
        'pred_r_std': ('pref', 'pred_rew_std'),
        'pred_r_std_member': ('pref', 'pred_rew_std_member'),
        # 'pred_r_coef_var': ('pref', 'pred_rew_coef_var'),
        'pred_r_uncertainty': ('pref', 'pred_rew_model_std'),
        # 'pred_r_uncertainty_coef_var': ('pref', 'pred_rew_model_coef_var'),
        'r': ('rollout', 'ep_rew_mean')
    }
    return LogRolloutStatsCallback(ep_info_log_keys)
