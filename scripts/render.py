import numpy as np
import gymnasium as gym
import shimmy
from relax import *
import jax
import jax.numpy as jnp
from relax.network.sdac import create_sdac_net
from relax.algorithm.sdac import SDAC
from relax.network.diffrep import create_diffrep_net
from relax.algorithm.diffrep import DiffRep
from relax.network.sac import create_sac_net
from relax.algorithm.sac import SAC
import yaml
from types import SimpleNamespace
import glob
import os
from functools import partial
import pickle
from tqdm import tqdm
import os.path as op
import datetime
import argparse

from moviepy import *

height, width = 480, 480
options = {'move_speed': 1.0} # ignored if not walker-multi

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder-path", type=str,
                       default="logs/dsac/walker-walk-v0/sdac_2025-03-02_00-45-08_s100_test_use_atp1")
argparser.add_argument("--num-simulations", type=int, default=10)
args = argparser.parse_args()
folder_path = args.folder_path
n_sims = args.num_simulations

file = sorted([(int(op.basename(file).split('-')[1]), file)
               for file in glob.glob(op.join(folder_path, 'policy-*.pkl'))], key=lambda el: el[0])[-1][1]
with open(file, "rb") as f:
    policy_params = pickle.load(f)

with open(folder_path + "/config.yaml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = SimpleNamespace(**args)

env = gym.make(args.env)
init_network_key = jax.random.PRNGKey(args.seed)
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
hidden_sizes = [args.hidden_dim] * args.hidden_num
diffusion_hidden_sizes = [
    args.diffusion_hidden_dim] * args.diffusion_hidden_num


def mish(x: jax.Array):
    return x * jnp.tanh(jax.nn.softplus(x))

if args.alg == 'sdac':
    agent, params = create_sdac_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=args.diffusion_steps,
                                        num_particles=args.num_particles,
                                        noise_scale=args.noise_scale,
                                        target_entropy_scale=args.target_entropy_scale)
    algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                        delay_alpha_update=args.delay_alpha_update,
                            lr_schedule_end=args.lr_schedule_end,
                            use_ema=args.use_ema_policy)
elif args.alg == 'diffrep':
    agent, params = create_diffrep_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=args.diffusion_steps, 
                                        num_particles=args.num_particles, 
                                        noise_scale=args.noise_scale,
                                        target_entropy_scale=args.target_entropy_scale)
    algorithm = DiffRep(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, 
                        delay_alpha_update=args.delay_alpha_update,
                            lr_schedule_end=args.lr_schedule_end,
                            use_ema=args.use_ema_policy, rep_weight=args.rep_weight)
elif args.alg == "sac":
    gelu = partial(jax.nn.gelu, approximate=False)
    agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
    algorithm = SAC(agent, params, lr=args.lr)
else:
    raise ValueError("Unknown algorithm")

@jax.jit
def get_action(obs):
    key = np.random.randint(0, 2**32-1)
    return algorithm.agent.get_action(jax.random.PRNGKey(key), policy_params, obs)

os.makedirs(op.join(folder_path, "videos"), exist_ok=True)
rewards = []
for _ in tqdm(range(n_sims)):
    obs, _ = env.reset(options=options)
    img = env.unwrapped.render(height=height, width=width)
    obss, _rewards, imgs, vels = [obs], [], [img], []
    while True:
        act = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        vels.append(info['current_speed'])
        imgs.append(env.unwrapped.render(height=height, width=width))
        obss.append(obs)
        _rewards.append(reward)
        if truncated or terminated:
            break
    rewards.append(sum(_rewards))
    clip = ImageSequenceClip(imgs, fps=round(1 / env.unwrapped.control_timestep))
    filename = op.join(folder_path, "videos", datetime.datetime.now().strftime("%m-%d_%H-%M-%S") + ".mp4")
    clip.write_videofile(filename)

    import matplotlib.pyplot as plt

    plt.plot(vels)
    target_speed = env.unwrapped.target_speed
    plt.plot([0, len(vels)], [target_speed, target_speed], 'r--')
    plt.grid()
    plt.ylabel("Speed", fontsize=16)
    plt.xlabel("Time", fontsize=16)
    plt.savefig(filename.replace(".mp4", ".png"))

print("Average episode return:", np.mean(rewards))
