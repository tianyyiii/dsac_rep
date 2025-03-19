from collections import deque
import random
import copy
import numpy as np
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box
    
class MetaWorldWrapper(Wrapper):
    def __init__(self, env, obs_type="state", n_stack=2, max_episode_steps=500):
        self.env = env
        self.obs_type = obs_type
        if self.obs_type == "image":
            self.observation_space = Box(low=0, high=255, shape=(64 * 64 * 3 * n_stack,), dtype=np.uint8)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.env._freeze_rand_vec = False
        self._max_episode_steps = max_episode_steps
        self._t = 0
        self.unwrapped.max_path_length = max_episode_steps

    def reset(self, **kwargs):
        obs = self.env.reset()
        self._t = 0
        if self.obs_type == "image":
            frame = self.env.render(offscreen=True, resolution=(64,64))
            for _ in range(self.n_stack):
                self.frames.append(frame)
            obs = np.concatenate(list(self.frames), axis=2)
            obs = obs.reshape(-1)
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
            obs = obs.reshape(-1)
        else:
            obs = obs.astype(np.float32)
        terminated = False
        truncated = (self._t >= self._max_episode_steps)
        return obs, total_reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped
    

class MultiTaskMetaWorldWrapper(Wrapper):
    def __init__(self, envs, obs_type="state", n_stack=2, max_episode_steps=500, pos_emb_dim=8):
        self.envs = envs
        self.env_num = len(self.envs)
        self.obs_type = obs_type
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        for env in self.envs:
            env._freeze_rand_vec = False
        self._max_episode_steps = max_episode_steps
        self._t = 0
        self.env_index = random.randint(0, self.env_num - 1)
        self.current_env = self.envs[self.env_index]
        if self.obs_type == "image":
            self.observation_space = Box(low=0, high=255, shape=(64 * 64 * 3 * n_stack,), dtype=np.uint8)
        else:
            orig_space = self.current_env.observation_space
            low = np.concatenate([orig_space.low, np.array([-np.inf]*pos_emb_dim)])
            high = np.concatenate([orig_space.high, np.array([np.inf]*pos_emb_dim)])
            self.observation_space = Box(low=low, high=high, dtype=orig_space.dtype)
        self.action_space = self.current_env.action_space
        self.positional_embeddings = self.get_sinusoid_encoding_table(self.env_num, pos_emb_dim)

    def get_sinusoid_encoding_table(self, n_position, d_hid):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
        return sinusoid_table

    def reset(self, env_index=None, **kwargs):
        if env_index:
            self.env_index = env_index
        else:
            self.env_index = random.randint(0, self.env_num - 1)
        self.current_env = self.envs[self.env_index]
        obs = self.current_env.reset()
        self._t = 0
        if self.obs_type == "image":
            frame = self.current_env.render(offscreen=True, resolution=(64,64))
            for _ in range(self.n_stack):
                self.frames.append(frame)
            obs = np.concatenate(list(self.frames), axis=2)
            obs = obs.reshape(-1)
        else:
            env_emb = self.positional_embeddings[self.env_index]
            env_emb = env_emb.astype(obs.dtype)
            obs = np.concatenate([obs, env_emb], axis=0)
        return obs, {}

    def step(self, action):
        total_reward = 0
        for _ in range(2):
            obs, reward, done, info = self.current_env.step(action.copy())
            if self.obs_type == "state":
                env_emb = self.positional_embeddings[self.env_index]
                env_emb = env_emb.astype(obs.dtype)
                obs = np.concatenate([obs, env_emb], axis=0)
            total_reward += reward
            self._t += 1
        if self.obs_type == "image":
            frame = self.current_env.render(offscreen=True, resolution=(64, 64))
            self.frames.append(frame)
            obs = np.concatenate(list(self.frames), axis=2)
            obs = obs.reshape(-1)
        else:
            obs = obs.astype(np.float32)
        terminated = False
        truncated = (self._t >= self._max_episode_steps)
        return obs, total_reward, terminated, truncated, info
    
    def close(self):
        for env in self.envs:
            env.close()

    @property
    def unwrapped(self):
        return self.current_env.unwrapped
    
'''
Only for state-based metaworld for now
'''
class MPCWrapper(Wrapper):
    def __init__(self, env, pred_horizon=1, act_horizon=1, gamma=0.99):
        self.env = env  
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.gamma = gamma

        self.observation_space = Box(
            low=np.tile(self.env.observation_space.low, pred_horizon),
            high=np.tile(self.env.observation_space.high, pred_horizon),
            dtype=self.env.observation_space.dtype
        )
        self.action_space = Box(
            low=np.tile(self.env.action_space.low, pred_horizon),
            high=np.tile(self.env.action_space.high, pred_horizon),
            dtype=self.env.action_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mpc_obs = np.concatenate([obs for _ in range(self.pred_horizon)], axis=0)
        return mpc_obs, info

    def step(self, actions):
        actions_seq = actions.reshape(self.pred_horizon, -1)

        total_discounted_reward = 0.0
        discount = self.gamma
        observations = []
        terminated = False
        truncated = False
        info = {}

        skip_pred = False
        for act in actions_seq[:self.act_horizon]:
            obs, reward, terminated, truncated, info = self.env.step(act.copy())
            total_discounted_reward += discount * reward
            discount *= self.gamma
            observations.append(obs)
            if terminated or truncated:
                skip_pred = True
                break
        cur_state = self.unwrapped.get_env_state()
        act_reward = total_discounted_reward
        
        if not skip_pred:
            for act in actions_seq[self.act_horizon:]:
                obs, reward, terminated, truncated, info = self.env.step(act.copy())
                total_discounted_reward += discount * reward
                discount *= self.gamma
                observations.append(obs)
                if terminated or truncated:
                    break
        self.unwrapped.set_env_state(cur_state)

        if len(observations) < self.pred_horizon:
            last_obs = observations[-1]
            for _ in range(self.pred_horizon - len(observations)):
                observations.append(last_obs)
            truncated = True

        mpc_obs = np.concatenate(observations, axis=0)
        info["act_reward"] = act_reward
        
        return mpc_obs, total_discounted_reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped