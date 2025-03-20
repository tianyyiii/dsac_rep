import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
from gymnasium.envs.registration import register
from typing import Any

class CartpoleEnv(gym.Env):
     
    observation_size = 4
    state_size = 4
    action_size = 1

    dt = 0.02
    force_mag = 10.0

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = 1.1
    length = 0.5 
    polemass_length = 0.6


    theta_threshold_radians = 12 *  np.pi / 180
    x_threshold = 2.4

    config_ref_name = "cartpole"

    def __init__(self, integrator='euler'):
        super().__init__()
        self.integrator = integrator
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    
    def generate_init_state(self, reset_noise_scale=5e-2):
        return (np.random.rand(self.state_size) * reset_noise_scale * 2 - reset_noise_scale).astype(np.float32)
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.state = self.generate_init_state()
        return self.state, {}

    def step(self, action):
        reward, terminated = self.compute_reward_and_done(self.state, action)
        self.state = self(self.state, action)
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def __call__(self, x, u, lib=np):
        th, vel, thdot = x[..., 1], x[..., 2], x[..., 3]
        
        force = self.force_mag * u[..., 0]
        costheta = lib.cos(th)
        sintheta = lib.sin(th)

        temp = (
            force + self.polemass_length * lib.square(thdot) * sintheta
        ) / self.total_mass
        thdotdot = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * lib.square(costheta) / self.total_mass)
        )
        acc = temp - self.polemass_length * thdotdot * costheta / self.total_mass

        if self.integrator == 'euler':
            x_dot = lib.stack([vel, thdot, acc, thdotdot], axis=-1)
        elif self.integrator == 'symplectic':
            x_dot = lib.stack([vel + acc * self.dt, thdot + thdotdot * self.dt, acc, thdotdot], axis=-1)
        else:
            raise Exception('Unknown integrator')
        return x + x_dot * self.dt # euler integration
    
    def dynamics(self, x, u):
        return self(x, u, lib=jnp)

    def compute_reward_and_done(self, x, u):
        healthy = (np.abs(x[0]) < self.x_threshold) & (np.abs(x[1]) < self.theta_threshold_radians)
        done = 1.0 - healthy.astype(np.float32)
        return 1.0, done
    

register(
    id=f'dsac/cartpole-v0',
    entry_point='relax.env.cartpole:CartpoleEnv',
    max_episode_steps=200,
)
