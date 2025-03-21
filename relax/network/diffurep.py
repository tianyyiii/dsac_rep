from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, URepPhiNet, URepMuNet, URepQNet, URepPolicyNet
from relax.network.common import WithSquashedGaussianPolicy
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data

class DiffURepParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_poicy: hk.Params
    phi: hk.Params
    mu: hk.Params
    log_alpha: jax.Array


@dataclass
class DiffURepNet:
    q: Callable[[hk.Params, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array], jax.Array]
    phi: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    mu: Callable[[hk.Params, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, self.noise_schedule)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, phi_params = policy_params

        def model_fn(t, x):
            def single_sample_jacobian(x_sample, obs_sample, t_sample):
                return jax.jacfwd(lambda x: self.phi(phi_params, obs_sample, x, t_sample))(x_sample)
            if len(obs.shape) == 2:
                if t.ndim == 0:
                    t = jnp.full((obs.shape[0],), t)
                feature_a = jax.vmap(single_sample_jacobian)(x, obs, t)
                feature_a = jnp.reshape(feature_a, (obs.shape[0], -1)) 
            else:
                feature_a = jax.jacfwd(lambda x: self.phi(phi_params, obs, x, t))(x)
                feature_a = jnp.reshape(feature_a, (-1,))
            return jax.lax.stop_gradient(self.policy(policy_params, feature_a))

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            feature_q = self.phi(phi_params, obs, act, 0)
            q1 = self.q(q1_params, feature_q)
            q2 = self.q(q2_params, feature_q)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act


    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params, phi_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params, phi_params)
        return self.get_action(key, policy_params, obs)
    

def create_diffurep_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feat_dim: int,
    hidden_sizes: Sequence[int],
    feat_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    ) -> Tuple[DiffURepNet, DiffURepParams]:

    q = hk.without_apply_rng(hk.transform(lambda feature: URepQNet(hidden_sizes, activation)(feature)))
    policy = hk.without_apply_rng(hk.transform(lambda feature: URepPolicyNet(hidden_sizes, activation, act_dim)(feature)))
    phi = hk.without_apply_rng(hk.transform(lambda obs, act, t: URepPhiNet(feat_hidden_sizes, activation, embedding_dim=feat_dim)(obs, act, t)))
    mu = hk.without_apply_rng(hk.transform(lambda next_obs: URepMuNet(feat_hidden_sizes, activation, embedding_dim=feat_dim)(next_obs)))

    @jax.jit
    def init(key, obs, act, feature, feature_a):
        q1_key, q2_key, policy_key, phi_key, mu_key = jax.random.split(key, 5)
        phi_params = phi.init(phi_key, obs, act, 0)
        q1_params = q.init(q1_key, feature)
        q2_params = q.init(q2_key, feature)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, feature_a)
        target_policy_params = policy_params
        mu_params = mu.init(mu_key, obs)
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return DiffURepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, phi_params, mu_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_feature = jnp.zeros((1, feat_dim))
    sample_feature_a = jnp.zeros((1, feat_dim * act_dim))
    params = init(key, sample_obs, sample_act, sample_feature, sample_feature_a)

    net = DiffURepNet(q=q.apply, policy=policy.apply, phi=phi.apply, mu=mu.apply, num_timesteps=num_timesteps, act_dim=act_dim, 
                    target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                    noise_schedule='linear')
    return net, params
