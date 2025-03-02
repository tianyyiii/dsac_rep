import gymnasium as gym
from .dog_wrapper import DogWrapper
from gymnasium.envs.registration import register

register(
    id='dsac/dog-walk-v0',
    entry_point='relax.env.dog.dog_wrapper:DogWrapper',
    max_episode_steps=1000,
    kwargs={'task_name': 'walk'},
)

register(
    id='dsac/dog-trot-v0',
    entry_point='relax.env.dog.dog_wrapper:DogWrapper',
    max_episode_steps=1000,
    kwargs={'task_name': 'trot'},
)

register(
    id='dsac/dog-run-v0',
    entry_point='relax.env.dog.dog_wrapper:DogWrapper',
    max_episode_steps=1000,
    kwargs={'task_name': 'run'},
)

register(
    id='dsac/dog-multi-v0',
    entry_point='relax.env.dog.dog_wrapper:DogWrapper',
    max_episode_steps=1000,
    kwargs={'task_name': 'multi'},
)