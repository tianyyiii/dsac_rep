import numpy as np
from gymnasium.envs.classic_control.pendulum import DEFAULT_X, DEFAULT_Y
import jax
import jax.numpy as jnp
from typing import Any
from gymnasium.envs.registration import register
import gymnasium as gym

class PendulumEnv(gym.Env):
     
    observation_size = 3
    state_size = 2
    action_size = 1

    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0

    config_ref_name = "pendulum"

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.state = self.generate_init_state()
        obs = self.get_obs(self.state)
        return obs, {}

    def step(self, action):
        reward = self.compute_reward(self.state, action)
        self.state = self(self.state, action)
        obs = self.get_obs(self.state)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def get_obs(self, state):
        th, thdot = state[..., 0], state[..., 1]
        return np.stack([np.cos(th), np.sin(th), thdot], axis=-1)
    
    def generate_init_state(self):
        high = np.array([DEFAULT_X, DEFAULT_Y / self.max_speed])
        return (np.random.rand(2) * high * 2 - high).astype('f')

    def __call__(self, x, u):
        th, thdot = x[..., 0], x[..., 1]
        m, l, g, dt = self.m, self.l, self.g, self.dt
        thdotdot = (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u[..., 0] * self.max_torque)
        newthdot = thdot +  thdotdot * dt / self.max_speed
        newthdot = np.clip(newthdot, -1, 1)
        x_plus = np.stack([th + newthdot * dt * self.max_speed, newthdot], axis=-1)
        return x_plus
    
    def dynamics(self, x, u):
        c, s, thdot = x[..., 0], x[..., 1], x[..., 2]
        m, l, g, dt = self.m, self.l, self.g, self.dt
        thdotdot = (3 * g / (2 * l) * s + 3.0 / (m * l**2) * u[..., 0] * self.max_torque)
        newthdot = thdot +  thdotdot * dt / self.max_speed
        newthdot = jnp.clip(newthdot, -1, 1)

        dth = newthdot * dt * self.max_speed
        sindth, cosdth = jnp.sin(dth), jnp.cos(dth)
        cnew = c * cosdth - s * sindth
        snew = s * cosdth + c * sindth
        return jnp.stack([cnew, snew, newthdot], axis=-1)
    
    def compute_reward(self, x, u):
        cost = angle_normalize(
            x[0]) ** 2 + 0.1 * (x[1] * self.max_speed)**2 + 0.001 * (self.max_torque * u[0])**2
        return -cost

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

register(
    id=f'dsac/pendulum-v0',
    entry_point='relax.env.pendulum:PendulumEnv',
    max_episode_steps=200,
)
