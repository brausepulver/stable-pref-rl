import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np
import shimmy.dm_control_compatibility
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


del shimmy.dm_control_compatibility
_HYPHI_GYM_ENVS_REGISTERED = False


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sample_obs = env.observation_space.sample()
        if isinstance(sample_obs, dict):
            total_size = sum(obs.size for obs in sample_obs.values())
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float64
            )

    def observation(self, obs):
        if isinstance(obs, dict):
            return np.concatenate([obs[k].ravel() for k in sorted(obs.keys())])
        return obs


def get_hyphi_gym_factory(name: str, record_video=False, render_mode=None):
    global _HYPHI_GYM_ENVS_REGISTERED

    from hyphi_gym import named, register_envs, Monitor as HyphiMonitor

    if not _HYPHI_GYM_ENVS_REGISTERED:
        register_envs()
        _HYPHI_GYM_ENVS_REGISTERED = True

    def factory() -> gym.Env:
        return HyphiMonitor(gym.make(render_mode=render_mode, **named(name)), record_video=record_video)

    return factory


def get_dm_control_factory(name: str, limit_ep_steps=1000, render_mode=None):
    def factory() -> gym.Env:
        env = gym.make(name, render_mode=render_mode)
        env = FlattenObservationWrapper(env)
        env = TimeLimit(env, limit_ep_steps)
        env = Monitor(env)
        return env

    return factory


def get_metaworld_factory(name: str, render_mode=None):
    def factory() -> gym.Env:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[name]
        env = env_cls(render_mode=render_mode)
        env._freeze_rand_vec = False
        env._set_task_called = True

        if env.spec is None:
            env.spec = gym.envs.registration.EnvSpec(id=name)

        env = TimeLimit(env, env.max_path_length)
        env = Monitor(env)
        return env

    return factory


def get_env_factory(name: str, **kwargs):
    """Create an environment."""

    if name.startswith("dm_control"):
        return get_dm_control_factory(name, **kwargs)

    if name.startswith("metaworld"):
        return get_metaworld_factory(name, **kwargs)

    return get_hyphi_gym_factory(name, **kwargs)


def make_vec_env(n_envs: int = 1, normalize: bool = False, **kwargs):
    """Create a vectorized environment."""

    env = DummyVecEnv([get_env_factory(**kwargs) for i in range(n_envs)])

    if normalize:
        env = VecNormalize(env, norm_reward=False)

    return env
