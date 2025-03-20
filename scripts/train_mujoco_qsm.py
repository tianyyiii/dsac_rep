import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml

import jax, jax.numpy as jnp
import wandb

from relax.algorithm.qsm import QSM
from relax.algorithm.qsm_rep import QSMRep
from relax.buffer import TreeBuffer
from relax.network.qsm import create_qsm_net
from relax.network.qsm_rep import create_qsm_rep_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair, RepresentationExperience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="qsm-rep")
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--suffix", type=str, default="test_use_atp1")
    parser.add_argument("--num_vec_envs", type=int, default=5)

    parser.add_argument("--feat_dim", type=int, default=2048)
    parser.add_argument("--feat_hidden_dim", type=int, default=1024)
    parser.add_argument("--feat_hidden_num", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_num", type=int, default=1)
    parser.add_argument("--score_hidden_dim", type=int, default=1024)
    parser.add_argument("--score_hidden_num", type=int, default=2)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_true_scores", default=False, action="store_true")
    
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=30_000) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=1_000_000) #1e6
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--target_entropy_scale", type=float, default=1.5)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--use_ema_policy", default=True, action="store_true")
    parser.add_argument("--use_wandb",default=False, action="store_true")
    parser.add_argument("--warmup_with", type=str, default="random")
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    score_hidden_sizes = [args.score_hidden_dim] * args.score_hidden_num
    feature_hidden_sizes = [args.feat_hidden_dim] * args.feat_hidden_num

    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == "qsm":
        assert not args.use_true_scores, "True scores are not supported in QSM"
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes,
                                       num_timesteps=args.diffusion_steps, num_particles=args.num_particles,
                                       target_entropy_scale=args.target_entropy_scale, noise_scale=args.noise_scale)
        algorithm = QSM(agent, params, lr=args.lr,
                        lr_schedule_end=args.lr_schedule_end, alpha_lr=args.alpha_lr)
    elif args.alg == "qsm-rep":
        agent, params = create_qsm_rep_net(init_network_key, obs_dim=obs_dim, act_dim=act_dim, feature_dim=args.feat_dim,
                                           score_hidden_sizes=score_hidden_sizes, hidden_sizes=hidden_sizes, 
                                           feature_hidden_sizes=feature_hidden_sizes, target_entropy_scale=args.target_entropy_scale,
                                           num_timesteps=args.diffusion_steps, num_particles=args.num_particles, noise_scale=args.noise_scale, use_true_scores=args.use_true_scores)
        algorithm = QSMRep(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, lr_schedule_end=args.lr_schedule_end, use_true_scores=args.use_true_scores)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    exp = Experience.create_example(obs_dim, act_dim, batch_size=1)
    buffer = TreeBuffer.from_experience(
        obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    trainer = OffPolicyTrainer(
        env=env,
        env_name=args.env,
        algorithm=algorithm,
        buffer=buffer,
        batch_size=args.batch_size,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        update_per_iteration=args.update_per_iteration,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 40),
        warmup_with=args.warmup_with,
        log_path=exp_dir,
    )

    if args.use_wandb:
        wandb.init(project='DiffRep', dir=exp_dir, sync_tensorboard=True)
        wandb.config.update(vars(args))
        wandb.config.update({'log_dir': exp_dir})
    
    trainer.setup(exp)
    log_git_details(log_file=os.path.join(exp_dir, 'git.diff'))

    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)

    if args.use_wandb:
        wandb.finish()