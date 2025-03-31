from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, Identity, fix_repr, dataclass
from functools import partial
from relax.network.common import WithSquashedDeterministicPolicy


class TD3Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params


@dataclass
class TD3Net(WithSquashedDeterministicPolicy):
    policy: Callable[[hk.Params, jax.Array], jax.Array]
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]


def create_td3_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int] = [512, 512, 512],
    exploration_noise: float = 0.2,
) -> Tuple[TD3Net, TD3Params]:
    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, activation=jax.nn.elu)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(
        lambda obs: mlp_w_norm(hidden_sizes, act_dim, activation=jax.nn.relu)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        target_policy = policy_params
        return TD3Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = TD3Net(policy=policy.apply, q=q.apply, exploration_noise=exploration_noise, preprocess=Identity)
    return net, params


@dataclass
@fix_repr
class QNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp_w_norm(self.hidden_sizes, 1, self.activation, squeeze_output=True)(input)


def mlp_w_norm(hidden_sizes: Sequence[int], output_size: int, activation: Activation, *, squeeze_output: bool = False) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size),  
                   hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True), 
                   activation]
    layers += [hk.Linear(output_size)]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return hk.Sequential(layers)
