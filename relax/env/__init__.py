from collections import deque
import random
import numpy as np
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv
from relax.env.env_wrapper import MetaWorldWrapper, MultiTaskMetaWorldWrapper, MPCWrapper


class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0):
        super().__init__(env)
        self.env: Env[np.ndarray, np.ndarray]

        # assert isinstance(env.observation_space, Box)
        # assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        assert env.action_space.is_bounded()
        if isinstance(env, VectorEnv):
            _, self.obs_dim = env.observation_space.shape
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_dim, = env.observation_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1,
            high=1,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    def reset(self, *, env_index=None, seed=None, options=None):
        if env_index:
            obs, info = self.env.reset(env_index=env_index, seed=seed, options=options)
        else:
            obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info

def create_env(name: str, seed: int, obs_type: str = "state", action_seed: int = 0, pred_horizon: int = 1, act_horizon: int = 1):
    if "metaworld" in name:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name.split("/")[1]](seed=seed)
        env = MetaWorldWrapper(env, obs_type)
    elif "mw-mpc" in name:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name.split("/")[1]](seed=seed)
        env = MetaWorldWrapper(env, obs_type, max_episode_steps=int(500 * pred_horizon / act_horizon + 1))   
        env = MPCWrapper(env, pred_horizon, act_horizon)    
    elif "mt-10" in name:
        env_names = [
        'window-open-v2-goal-observable',
        'window-close-v2-goal-observable',
        'peg-insert-side-v2-goal-observable',
        'door-open-v2-goal-observable',
        'drawer-open-v2-goal-observable',
        'pick-place-v2-goal-observable',
        'reach-v2-goal-observable',
        'button-press-topdown-v2-goal-observable',
        'drawer-close-v2-goal-observable',
        'push-v2-goal-observable',
        ]
        envs = [ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed) for env_name in env_names]
        env = MultiTaskMetaWorldWrapper(envs, obs_type)
    else:
        env = make(name)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, obs_type: str = "state", mode: str = "serial", pred_horizon:int = 1, act_horizon: int = 1, **kwargs):
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    env = Impl(name, num_envs, seed, obs_type, pred_horizon, act_horizon, **kwargs)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
