from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet, FeatureNet, mlp
from relax.network.sdac import SDACNet, Diffv2Params
from relax.network.sdac_rep import SDACRepNet
from relax.utils.jax_utils import random_key_from_data
from typing import Union


class SDACRandRepParams(NamedTuple):
    # old
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array
    # new
    feature: hk.Params
    target_feature: hk.Params
    theta: hk.Params

def create_sdac_rand_rep_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feature_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    feature_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale=0.9,
) -> Tuple[SDACNet, Diffv2Params]:

    def Identity(x): return x

    q = hk.without_apply_rng(hk.transform(lambda feature: mlp(
        hidden_sizes, 1, activation, output_activation=Identity, squeeze_output=True)(feature)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(
        diffusion_hidden_sizes, activation)(obs, act, t)))

    feature = hk.without_apply_rng(hk.transform(lambda obs, act: FeatureNet(
        feature_hidden_sizes, feature_dim, activation)(obs, act)))
    
    key, kw, kb = jax.random.split(key, 3)
    w = jax.random.normal(kw, (obs_dim, feature_dim))
    b = jax.random.uniform(kb, (feature_dim)) * 2 * jnp.pi

    def mu(next_obs):
        return jnp.cos(jnp.squeeze(next_obs[..., None, :] @ w, axis=-2) + b)

    theta = hk.without_apply_rng(hk.transform(
        lambda feature: hk.Linear(1)(feature).squeeze(-1)))

    @jax.jit
    def init(key, obs, act, feat):
        q1_key, q2_key, policy_key, feature_key, mu_key, theta_key = jax.random.split(
            key, 6)
        q1_params = q.init(q1_key, feat)
        q2_params = q.init(q2_key, feat)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params

        feature_params = feature.init(feature_key, obs, act)
        target_feature_params = feature_params
        theta_params = theta.init(theta_key, feat)

        # math.log(3) or math.log(5) choose one
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return SDACRandRepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, 
                                 target_policy_params, log_alpha, feature_params, target_feature_params, 
                                 theta_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_feat = jnp.zeros((1, feature_dim))
    params = init(key, sample_obs, sample_act, sample_feat)

    net = SDACRepNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim,
                     target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                     noise_schedule='cosine', feature=feature.apply, mu=mu, theta=theta.apply)
    return net, params
