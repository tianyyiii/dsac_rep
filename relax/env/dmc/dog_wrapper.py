import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_control.suite.dog import *
from dm_control.suite.dog import _RUN_SPEED, _WALK_SPEED, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP
from dm_env import StepType
from typing import Any


class DogWrapper(gym.Env):
    def __init__(self, task_name='walk'):
        self._env = suite.load(domain_name="dog", task_name=task_name)
        self._task_name = task_name
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._env.action_spec().shape[0],), dtype=np.float32)
        obs_dim = sum([v.shape[0]
                      for v in self._env.observation_spec().values()])
        if task_name == 'multi':
            obs_dim += 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
    @property
    def move_speed(self):
        return self._env.task._move_speed

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None,):
        if self._task_name == 'multi':
            if options is not None and 'move_speed' in options:
                move_speed = options['move_speed']
            else:
                random = np.random.RandomState(seed)
                move_speed = (_RUN_SPEED - _WALK_SPEED) * random.uniform() + _WALK_SPEED
            self._env.task._move_speed = move_speed

        obs = self._env.reset()
        obs = self.extract_obs(obs)
        info = {'move_speed': self.move_speed}
        return obs, info

    def step(self, action):
        timestep = self._env.step(action)
        obs = self.extract_obs(timestep)
        terminated = timestep.step_type == StepType.LAST
        truncated = False
        info = {'move_speed': self.move_speed}
        return obs, timestep.reward, terminated, truncated, info

    def extract_obs(self, timestep):
        obs = np.concatenate(list(timestep.observation.values()))
        if self._task_name == 'multi':
            obs = np.append(obs, self.move_speed)
        return obs.astype('f')


@SUITE.add('no_reward_visualization')
def multi(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  if not isinstance(random, np.random.RandomState):
      random = np.random.RandomState(random)
  move_speed = (_RUN_SPEED - _WALK_SPEED) * random.uniform() + _WALK_SPEED
  floor_size = move_speed * _DEFAULT_TIME_LIMIT
  physics = Physics.from_xml_string(*get_model_and_assets(floor_size))
  task = Move(move_speed=move_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)
