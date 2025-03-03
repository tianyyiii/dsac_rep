import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_control.suite.walker import *
from dm_control.suite.walker import _RUN_SPEED, _WALK_SPEED, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP
from dm_env import StepType
from typing import Any


class WalkerWrapper(gym.Env):
    control_timestep: float = _CONTROL_TIMESTEP
    def __init__(self, task_name='walk'):
        self._env = suite.load(domain_name="walker", task_name=task_name)
        self._task_name = task_name
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._env.action_spec().shape[0],), dtype=np.float32)
        obs_dim = sum([(v.shape[0] if len(v.shape) > 0 else 1)
                      for v in self._env.observation_spec().values()])
        if task_name == 'multi':
            obs_dim += 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
    @property
    def target_speed(self):
        return self._env.task._move_speed

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None,):
        if self._task_name == 'multi':
            if options is not None and 'target_speed' in options:
                target_speed = options['target_speed']
            else:
                random = np.random.RandomState(seed)
                target_speed = (_RUN_SPEED - _WALK_SPEED) * random.uniform() + _WALK_SPEED
            self._env.task._move_speed = target_speed

        obs = self._env.reset()
        obs = self.extract_obs(obs)
        info = {'current_speed': 0.0}
        return obs, info

    def step(self, action):
        timestep = self._env.step(action)
        obs = self.extract_obs(timestep)
        terminated = timestep.step_type == StepType.LAST
        truncated = False
        info = {'current_speed': self._env.physics.horizontal_velocity()}
        return obs, timestep.reward, terminated, truncated, info

    def extract_obs(self, timestep):
        obs = np.concatenate(
            list([[v] if isinstance(v, float) else v for v in timestep.observation.values()]))
        if self._task_name == 'multi':
            obs = np.append(obs, self.target_speed)
        return obs.astype('f')
    
    def render(self, height=480, width=480, camera_id=0):
        return self._env.physics.render(height, width, camera_id=camera_id)

@SUITE.add('benchmarking')
def multi(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  if not isinstance(random, np.random.RandomState):
      random = np.random.RandomState(random)
  target_speed = (_RUN_SPEED - _WALK_SPEED) * random.uniform() + _WALK_SPEED
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PlanarWalker(move_speed=target_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
