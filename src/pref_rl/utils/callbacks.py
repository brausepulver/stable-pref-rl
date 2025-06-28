from .logging import LogRolloutStatsCallback


def get_default_callbacks():
    ep_info_log_keys = {
        'intr_r': ('pretrain', 'ep_intr_rew_mean'),
        'pred_r_mean': ('pref', 'ep_rew_mean'),
        'pred_r_step': ('pref', 'step_rew_mean'),
        'pred_r_std': ('pref', 'ep_rew_std'),
        'pred_r_std_member': ('reward_model', 'avg_member_std_ep'),
        # 'pred_r_coef_var': ('reward_model', 'pred_rew_coef_var'),
        'pred_r_uncertainty': ('reward_model', 'avg_ensemble_std_rew'),
        # 'pred_r_uncertainty_coef_var': ('reward_model/', 'rew_model_coef_var'),
        'r': ('rollout', 'ep_rew_mean_current')
    }
    return LogRolloutStatsCallback(ep_info_log_keys)
