from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, FeatureNet, mlp, UDiffRepDPhiDaNet, UDiffRepPhiNet, UDiffRepPolicyNet
from relax.network.sdac import SDACNet, Diffv2Params
from relax.utils.jax_utils import random_key_from_data
from typing import Union

class UDiffRepParams(NamedTuple):
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
    mu: hk.Params

@dataclass
class UDiffRepNet(SDACNet):
    # old
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str
    # new
    feature: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    mu: Callable[[hk.Params, jax.Array], jax.Array]

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, feat_params, _ = policy_params
        obs_ndim = obs.ndim
        obs = jnp.atleast_2d(obs)

        def model_fn(t, x):
            d_phi_d_a = jax.vmap(jax.jacfwd(
                lambda _x, _obs:  self.feature(feat_params, _obs, _x, t)))(x, obs)
            return self.policy(policy_params, d_phi_d_a)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(
                key, model_fn, (*obs.shape[:-1], self.act_dim))
            feats = self.feature(feat_params, obs, act, jnp.zeros(obs.shape[0]))
            q1 = self.q(q1_params, feats)
            q2 = self.q(q2_params, feats)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(
                acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * \
            jnp.exp(log_alpha) * self.noise_scale
        if obs_ndim == 1:
            act = act.squeeze(axis=0)
        return act

    def get_batch_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, q_func: Callable) -> jax.Array:
        raise NotImplementedError("Not implemented yet")

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params = (policy_params[0], -jnp.inf, *policy_params[2:])
        return self.get_action(key, policy_params, obs)

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError("Not implemented yet")

def create_sdac_rep_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feature_dim: int,
    hidden_sizes: Sequence[int],
    feature_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    ) -> Tuple[SDACNet, Diffv2Params]:
    
    Identity = lambda x: x
    
    q = hk.without_apply_rng(hk.transform(lambda feat: mlp(hidden_sizes, 1, activation, output_activation=Identity, squeeze_output=True)(feat)))
    policy = hk.without_apply_rng(hk.transform(lambda feat: UDiffRepPolicyNet(hidden_sizes, act_dim, activation)(feat)))
    feature = hk.without_apply_rng(hk.transform(lambda obs, act, t: UDiffRepPhiNet(feature_hidden_sizes, feature_dim, activation)(obs, act, t)))
    mu = hk.without_apply_rng(hk.transform(lambda obs: mlp(
        feature_hidden_sizes, feature_dim, activation, output_activation=jax.nn.tanh)(obs)))

    @jax.jit
    def init(key, obs, act, feat, d_feat_d_a):
        q1_key, q2_key, policy_key, feature_key, mu_key = jax.random.split(key, 5)
        
        q1_params = q.init(q1_key, feat)
        q2_params = q.init(q2_key, feat)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, d_feat_d_a)
        target_policy_params = policy_params
        
        feature_params = feature.init(feature_key, obs, act, jnp.zeros(1))
        target_feature_params = feature_params
        mu_params = mu.init(mu_key, obs)

        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return UDiffRepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, 
                             target_policy_params, log_alpha, feature_params, target_feature_params, 
                             mu_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_feat = jnp.zeros((1, feature_dim))
    sample_d_feat_d_a = jnp.zeros((1, feature_dim, act_dim))
    params = init(key, sample_obs, sample_act, sample_feat, sample_d_feat_d_a)

    net = UDiffRepNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim,
                    target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                    noise_schedule='linear', feature=feature.apply, mu=mu.apply)
    return net, params
