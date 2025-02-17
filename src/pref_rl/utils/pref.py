from collections import deque
import einops
import numpy as np
import torch
import torch.nn.functional as F


class EpisodeBuffer:
    def __init__(self, num_envs, num_episodes):
        self._env_buffer = [[] for _ in range(num_envs)]
        self.episodes = deque(maxlen=num_episodes)


    def add(self, value: torch.Tensor, done: np.ndarray):
        for env_idx, env_value in enumerate(value):
            self._env_buffer[env_idx].append(env_value)

        for env_idx in np.argwhere(done).reshape(-1):
            episode = self._env_buffer[env_idx]
            self.episodes.append(torch.stack(episode))
            episode.clear()


class Teacher:
    def __init__(self, segment_size: int, observation_size: int, action_size: int, teacher: str = None, teacher_kwargs: dict = None):
        teacher_kwargs = teacher_kwargs or {}
        self.segment_size = segment_size
        self.observation_size = observation_size
        self.action_size = action_size

        self.beta = teacher_kwargs.get('beta', 1 if teacher == 'stoc' else float('inf'))
        self.gamma = teacher_kwargs.get('gamma', 0.9 if teacher == 'myopic' else 1)
        self.eps_mistake = teacher_kwargs.get('eps_mistake', 0.1 if teacher == 'mistake' else 0)

        eps_adapt = 0.1
        self.eps_equal = teacher_kwargs.get('eps_equal', eps_adapt if teacher == 'equal' else 0)
        self.eps_skip = teacher_kwargs.get('eps_skip', eps_adapt if teacher == 'skip' else 0)
        self.threshold_equal = 0
        self.threshold_skip = 0


    def update_thresholds(self, episodes):
        _, gt_rewards = torch.split(torch.stack(episodes), (self.observation_size + self.action_size, 1), dim=-1)
        ep_lens, ep_rets = zip(*[(len(ep_rewards), sum(ep_rewards)) for ep_rewards in gt_rewards])

        margin = np.mean(ep_rets) * self.segment_size / np.mean(ep_lens)
        self.threshold_equal = margin * self.eps_equal
        self.threshold_skip = margin * self.eps_skip


    def query_segments(self, gt_rewards: torch.Tensor):
        myopia_exp = torch.arange(self.segment_size - 1, -1, -1)
        myopia_coef = torch.tensor(self.gamma).pow(myopia_exp)
        myopic_rewards = myopia_coef * gt_rewards

        returns = myopic_rewards.sum(dim=-1)
        left_ret, right_ret = returns

        if self.beta == float('inf'):
            preferences = (left_ret < right_ret).to(dtype=torch.long)
        else:
            probabilities = F.softmax(returns, dim=0)
            preferences = torch.bernoulli(probabilities[0]).to(dtype=torch.long)

        equal_indices = torch.argwhere(torch.abs(left_ret - right_ret) < self.threshold_equal).squeeze(-1)
        preferences[equal_indices] = 0.5

        keep_indices = torch.argwhere(torch.max(torch.abs(returns) >= self.threshold_skip, dim=0).values).squeeze(-1)
        preferences = preferences[keep_indices]
        return preferences, keep_indices


class Sampler:
    def __init__(self, segment_size: int, observation_size: int, action_size: int, pre_sample_multiplier: int = 10):
        self.segment_size = segment_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.pre_sample_multiplier = pre_sample_multiplier


    def sample_segments(self, episodes: list, num_samples: int, method: str = 'uniform', reward_model: callable = None):
        assert method in ('uniform', 'disagreement', 'entropy'), f"Unknown sampling method: {method}"

        valid_episodes = [ep for ep in episodes if len(ep) >= self.segment_size]

        num_samples_expanded = num_samples if method == 'uniform' else self.pre_sample_multiplier * num_samples
        ep_indices = torch.randint(0, len(valid_episodes), (2 * num_samples_expanded,))

        segments = []
        for ep_idx in ep_indices:
            ep = valid_episodes[ep_idx]

            start_step = np.random.randint(0, len(ep) - self.segment_size)
            offsets = torch.arange(0, self.segment_size)
            step_indices = start_step + offsets

            segment = ep[step_indices]
            segments.append(segment)

        obs, act, gt_rewards = torch.split(torch.stack(segments), (self.observation_size, self.action_size, 1), dim=-1)
        state_actions = torch.cat([obs, act], dim=-1)

        split_state_actions = einops.rearrange(state_actions, '(n p) s d -> n p s d', p=2)
        split_rewards = einops.rearrange(gt_rewards, '(n p) s 1 -> p n s', p=2)

        if method == 'uniform':
            return split_state_actions, split_rewards

        with torch.no_grad():
            member_rewards = reward_model(split_state_actions)
            member_returns = einops.reduce(member_rewards, 'm n p s 1 -> m n p', 'sum')
            probabilities = F.softmax(member_returns, dim=-1)

            if method == 'disagreement':
                metric = probabilities[..., 0].std(dim=0)
            elif method == 'entropy':
                entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
                metric = entropy.mean(dim=0)

            idx = torch.topk(metric, num_samples).indices
            return split_state_actions[idx], split_rewards[:, idx]
