import gymnasium as gym
from .dog_wrapper import DogWrapper
from .walker_wrapper import WalkerWrapper
from gymnasium.envs.registration import register

for task_name in ['walk', 'trot', 'run', 'multi']:
    register(
        id=f'dsac/dog-{task_name}-v0',
        entry_point='relax.env.dmc.dog_wrapper:DogWrapper',
        max_episode_steps=1000,
        kwargs={'task_name': task_name},
    )

for task_name in ['walk', 'run', 'multi']:
    register(
        id=f'dsac/walker-{task_name}-v0',
        entry_point='relax.env.dmc.walker_wrapper:WalkerWrapper',
        max_episode_steps=1000,
        kwargs={'task_name': task_name},
    )
