from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, QNet, QScoreNet, mlp, Identity, FeatureNet
from relax.utils.langevin import LangevinDynamics


class QSMRepParams(NamedTuple):
    # old
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    q_score: hk.Params
    log_alpha: jax.Array
    # new
    feature: hk.Params
    target_feature: hk.Params
    mu: hk.Params
    theta: hk.Params


@dataclass
class QSMRepNet:
    # old
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    q_score: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    # new
    feature: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    mu: Callable[[hk.Params, jax.Array], jax.Array]
    theta: Callable[[hk.Params, jax.Array], jax.Array]
    target_entropy: float
    noise_scale: float
    use_true_scores: bool

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, *, num_particles: Optional[int] = None) -> jax.Array:
        langevin = LangevinDynamics(self.num_timesteps)
        score_params, log_alpha, q1_params, q2_params, feat_params, _, _ = policy_params
        def model_fn(x):
            if self.use_true_scores:
                def q(x):
                    feats = self.feature(feat_params, obs, x)
                    q1 = self.q(q1_params, feats)
                    q2 = self.q(q2_params, feats)
                    return jnp.minimum(q1, q2).sum()
                return jax.grad(q)(x)
            else:
                return self.q_score(score_params, obs, x)
                

        def sample(key):
            act = langevin.sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            feats = self.feature(feat_params, obs, act)
            q1 = self.q(q1_params, feats)
            q2 = self.q(q2_params, feats)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        num_particles = num_particles if num_particles is not None else self.num_particles
        key, noise_key = jax.random.split(key)
        assert num_particles > 0
        if num_particles == 1:
            act = langevin.sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        else:
            keys = jax.random.split(key, num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array, *, num_particles: Optional[int] = None) -> jax.Array:
        # NOTE: Not sure if it is wise to get deterministic action from the score model
        key = jax.random.key(0)
        policy_params = (policy_params[0], -jnp.inf, *policy_params[2:])
        return self.get_action(key, policy_params, obs, num_particles=num_particles)

    def get_q_score_from_gradient(self, feat_params: hk.Params, q_params: hk.Params, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        def inner(act: jax.Array, obs: jax.Array):
            # NOTE: if a special q network cannot handle unbatched inputs,
            #       we can manually unsqueeze & squeeze here
            feats = self.feature(feat_params, obs, act)
            return self.q(q_params, feats)
        return jax.vmap(jax.value_and_grad(inner))(act, obs)

def create_qsm_rep_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    feature_dim: int,
    hidden_sizes: Sequence[int],
    score_hidden_sizes: Sequence[int],
    feature_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 100,
    num_particles: int = 1,
    target_entropy_scale: float = 0.9,
    noise_scale: float = 0.1,
    use_true_scores: bool = False
) -> Tuple[QSMRepNet, QSMRepParams]:
    q = hk.without_apply_rng(hk.transform(lambda feature: mlp(
        hidden_sizes, 1, activation, output_activation=Identity, squeeze_output=True)(feature)))
    q_score = hk.without_apply_rng(hk.transform(
        lambda obs, act: QScoreNet(score_hidden_sizes, activation)(obs, act)))

    feature = hk.without_apply_rng(hk.transform(lambda obs, act: FeatureNet(
        feature_hidden_sizes, feature_dim, activation)(obs, act)))
    
    mu = hk.without_apply_rng(hk.transform(lambda obs: mlp(
        feature_hidden_sizes, feature_dim, activation, output_activation=jax.nn.tanh)(obs)))

    theta = hk.without_apply_rng(hk.transform(
        lambda feature: hk.Linear(1)(feature).squeeze(-1)))

    @jax.jit
    def init(key, obs, act, feat):
        q1_key, q2_key, q_score_key, feature_key, mu_key, theta_key = jax.random.split(key, 6)
        q1_params = q.init(q1_key, feat)
        q2_params = q.init(q2_key, feat)
        target_q1_params = q1_params
        target_q2_params = q2_params
        q_score_params = q_score.init(q_score_key, obs, act)

        feature_params = feature.init(feature_key, obs, act)
        target_feature_params = feature_params
        mu_params = mu.init(mu_key, obs)
        theta_params = theta.init(theta_key, feat)
        # math.log(3) or math.log(5) choose one
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)

        return QSMRepParams(q1_params, q2_params, target_q1_params, target_q2_params, q_score_params, log_alpha,
                            feature_params, target_feature_params, mu_params, theta_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_feat = jnp.zeros((1, feature_dim))
    params = init(key, sample_obs, sample_act, sample_feat)

    net = QSMRepNet(q=q.apply, q_score=q_score.apply, num_timesteps=num_timesteps, 
                    act_dim=act_dim, num_particles=num_particles, feature=feature.apply, mu=mu.apply, 
                    theta=theta.apply, target_entropy=-act_dim*target_entropy_scale, noise_scale=noise_scale, use_true_scores=use_true_scores)
    return net, params
