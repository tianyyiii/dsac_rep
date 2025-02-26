from collections import deque
import numpy as np
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv

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

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info
    
class MetaWorldWrapper(Wrapper):
    def __init__(self, env, obs_type="state", n_stack=2, max_episode_steps=500):
        self.env = env
        self.obs_type = obs_type
        if self.obs_type == "image":
            self.observation_space = Box(low=0, high=255, shape=(64, 64, 3 * n_stack), dtype=np.uint8)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.env._freeze_rand_vec = False
        self._max_episode_steps = max_episode_steps
        self._t = 0

    def reset(self, **kwargs):
        obs = self.env.reset()
        self._t = 0
        if self.obs_type == "image":
            frame = self.env.render(offscreen=True, resolution=(64,64))
            for _ in range(self.n_stack):
                self.frames.append(frame)
            obs = np.concatenate(list(self.frames), axis=2)
        return obs, {}

    def step(self, action):
        total_reward = 0
        for _ in range(2):
            obs, reward, done, info = self.env.step(action.copy())
            total_reward += reward
            self._t += 1
        if self.obs_type == "image":
            frame = self.env.render(offscreen=True, resolution=(64, 64))
            self.frames.append(frame)
            obs = np.concatenate(list(self.frames), axis=2)
        else:
            obs = obs.astype(np.float32)
        terminated = False
        truncated = (self._t >= self._max_episode_steps)
        return obs, total_reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped
    

def create_env(name: str, seed: int, obs_type: str = "state", action_seed: int = 0):
    if "metaworld" in name:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name.split("/")[1]](seed=seed)
        env = MetaWorldWrapper(env, obs_type)
    else:
        env = make(name)
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, obs_type: str = "state", mode: str = "serial", **kwargs):
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    env = Impl(name, num_envs, seed, **kwargs)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
