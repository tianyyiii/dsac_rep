from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math
from .blocks import Identity, fix_repr

from relax.network.blocks import Activation, DACERPolicyNet, mlp, scaled_sinusoidal_encoding
from relax.network.sdac import SDACNet, Diffv2Params
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data
from typing import Union
from relax.network.sdac_rep import SDACRepNet

class SDACDiffSRParams(NamedTuple):
    # old
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: hk.Params
    # feat
    feature: hk.Params
    target_feature: hk.Params
    mu: hk.Params
    target_mu: hk.Params
    theta: hk.Params
    target_theta: hk.Params


@dataclass
class SDACDiffSRNet(SDACRepNet):

    feature_num_timesteps: int
    feature_noise_schedule: str
    
    @property
    def feature_diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.feature_num_timesteps, self.feature_noise_schedule)


def create_sdac_diffsr_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feature_dim: int,
    hidden_dim: int,
    diffusion_hidden_sizes: Sequence[int],
    feat_hidden_dim: int = 256,
    feat_n_blocks: int = 2,
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    feature_num_timesteps: int = 50,
    ) -> Tuple[SDACNet, Diffv2Params]:
    
    Identity = lambda x: x
    
    q = hk.without_apply_rng(hk.transform(lambda feature: RFFCritic(hidden_dim)(feature)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    feature = hk.without_apply_rng(hk.transform(lambda obs, act: FeatureNet(feature_dim, feat_hidden_dim, feat_n_blocks)(obs, act)))
    
    mu = hk.without_apply_rng(hk.transform(lambda obs, t: MuNet(
        feature_dim, obs_dim, feat_hidden_dim, feat_n_blocks)(obs, t)))
    
    theta = hk.without_apply_rng(hk.transform(lambda feature: ThetaNet(512)(feature))) # hardcoded
    
    @jax.jit
    def init(key, obs, act, feat):
        q1_key, q2_key, policy_key, feature_key, mu_key, theta_key = jax.random.split(key, 6)
        q1_params = q.init(q1_key, feat)
        q2_params = q.init(q2_key, feat)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        
        feature_params = feature.init(feature_key, obs, act)
        target_feature_params = feature_params
        mu_params = mu.init(mu_key, obs, 0)
        target_mu_params = mu_params
        theta_params = theta.init(theta_key, feat)
        target_theta_params = theta_params

        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return SDACDiffSRParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params,
                                target_policy_params, log_alpha, feature_params, target_feature_params, 
                                mu_params, target_mu_params, theta_params, target_theta_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_feat = jnp.zeros((1, feature_dim))
    params = init(key, sample_obs, sample_act, sample_feat)

    net = SDACDiffSRNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim,
                        target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                        noise_schedule='cosine', feature_num_timesteps=feature_num_timesteps, feature_noise_schedule='vp', 
                        feature=feature.apply, mu=mu.apply, theta=theta.apply)
    return net, params


@dataclass
@fix_repr
class RFFCritic(hk.Module):
    hidden_dim: int

    def __call__(self, feat):
        out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(feat)
        out = RFFLayer(self.hidden_dim)(out)
        out = hk.Linear(self.hidden_dim)(out)
        out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(out)
        out = jax.nn.elu(out)
        return hk.Linear(1)(out).squeeze(-1)
    

@dataclass
@fix_repr
class FeatureNet(hk.Module):
    feat_dim: int
    hidden_dim: int
    num_blocks: int
    embed_dim: int = 128

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        obs = hk.Linear(self.embed_dim * 2)(obs)
        obs = jax.nn.mish(obs)
        obs = hk.Linear(self.embed_dim)(obs)
        
        act = hk.Linear(self.embed_dim * 2)(act)
        act = jax.nn.mish(act)
        act = hk.Linear(self.embed_dim)(act)

        xu = jnp.concatenate((obs, act), axis=-1)
        return MLPResNet(self.hidden_dim, self.num_blocks, self.feat_dim)(xu)


@dataclass
@fix_repr
class MuNet(hk.Module):
    feat_dim: int
    state_dim: int
    hidden_dim: int
    num_blocks: int
    embed_dim: int = 128

    def __call__(self, obs: jax.Array, t: jax.Array) -> jax.Array:
        t_emb = scaled_sinusoidal_encoding(t, dim=self.embed_dim, batch_shape=obs.shape[:-1])
        inp  = jnp.concatenate((obs, t_emb), axis=-1)
        out = MLPResNet(self.hidden_dim, self.num_blocks, self.feat_dim * self.state_dim)(inp)
        return out.reshape(-1, self.feat_dim, self.state_dim)


@dataclass
@fix_repr
class ThetaNet(hk.Module):
    hidden_dim: int

    def __call__(self, feature: jax.Array) -> jax.Array:
        out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(feature)
        out = RFFLayer(self.hidden_dim)(out)
        out = hk.Linear(self.hidden_dim)(out)
        out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(out)
        out = jax.nn.elu(out)
        return hk.Linear(1)(out).squeeze(-1)

@dataclass
@fix_repr   
class RFFLayer(hk.Module):
    hidden_dim: int

    def __call__(self, x: jax.Array) -> jax.Array:
        x = hk.Linear(self.hidden_dim)(x)
        return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    

@dataclass
@fix_repr
class MLPResNetBlock(hk.Module):
    dim: int

    def __call__(self, x):
        residual = x
        x = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)(x)
        x = hk.Linear(self.dim * 4)(x)
        x = jax.nn.mish(x)
        x = hk.Linear(self.dim)(x)

        residual = hk.Linear(self.dim)(residual)

        return residual + x
    

@dataclass
@fix_repr
class MLPResNet(hk.Module):
    hidden_dim: int
    num_blocks: int
    out_dim: int

    def __call__(self, x):
        x = hk.Linear(self.hidden_dim)(x)

        for i in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim)(x)

        x = jax.nn.mish(x)
        x = hk.Linear(self.out_dim)(x)

        return x


