import torch
from .reward_mod import RewardModifierCallback


class KLPenaltyCallback(RewardModifierCallback):
    def __init__(self, pref_callback, kl_penalty_coef: float = 0.1, **kwargs):
        super().__init__(ep_info_key='kl_penalty_r', log_prefix='kl_penalty/', **kwargs)
        self.pref_callback = pref_callback
        self.kl_penalty_coef = kl_penalty_coef
        self.reference_policy_params = None

    def _predict_rewards(self):
        # Store reference policy when RM training finishes
        if (self.pref_callback._is_done_training() and self.reference_policy_params is None):
            # Create a brand new policy instance
            self.reference_policy = self.model.policy_class(
                observation_space=self.model.observation_space,
                action_space=self.model.action_space,
                lr_schedule=lambda x: 0,  # Dummy schedule
                use_sde=self.model.use_sde,
                **self.model.policy_kwargs
            ).to(self.model.device)
            
            # Load the current state
            self.reference_policy.load_state_dict(self.model.policy.state_dict())
            self.reference_policy.eval()
            
            # Don't store params anymore, just use the policy directly
            self.reference_policy_params = True  # Just a flag
            
        current_rewards = torch.tensor(self.locals['rewards'], dtype=torch.float)
        
        if self.reference_policy_params is None:
            return current_rewards
            
        obs, act, _ = self._get_current_step()
        with torch.no_grad():
            _, current_log_prob, _ = self.model.policy.evaluate_actions(obs, act)
            _, ref_log_prob, _ = self.reference_policy.evaluate_actions(obs, act)
            
        kl_div = current_log_prob - ref_log_prob
        kl_penalty = -self.kl_penalty_coef * kl_div.squeeze()
        
        return current_rewards + kl_penalty
