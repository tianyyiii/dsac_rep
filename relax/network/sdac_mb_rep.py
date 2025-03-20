from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet, MBCriticNet
from relax.network.sdac import SDACNet, Diffv2Params
from relax.utils.jax_utils import random_key_from_data
from typing import Union

@dataclass
class SDACMBRepNet(SDACNet):
    # old
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    dynamics: Callable[[jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params  = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)
        obs_ndim = obs.ndim
        obs = jnp.atleast_2d(obs)
        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(
                key, model_fn, (*obs.shape[:-1], self.act_dim))
            next_obs = self.dynamics(obs, act)
            q1, _ = self.q(*q1_params, next_obs)
            q2, _ = self.q(*q2_params, next_obs)
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

def create_sdac_mb_rep_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feature_dim: int,
    dynamics: Callable,
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    ) -> Tuple[SDACNet, Diffv2Params]:

    q = hk.without_apply_rng(hk.transform_with_state(lambda obs: MBCriticNet(feature_dim)(obs)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs)
        q2_params = q.init(q2_key, obs)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, 
                            target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = SDACMBRepNet(q=q.apply, policy=policy.apply, dynamics=dynamics, num_timesteps=num_timesteps, act_dim=act_dim,
                       target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                       noise_schedule='linear')
    return net, params
