from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.qsm_rep import QSMRepNet, QSMRepParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class QSMRepOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    q_score: optax.OptState
    log_alpha: optax.OptState
    feature: optax.OptState


class QSMRepTrainState(NamedTuple):
    params: QSMRepParams
    opt_state: QSMRepOptStates
    step: int


class QSMRep(Algorithm):
    def __init__(self, 
                 agent: QSMRepNet, 
                 params: QSMRepParams, 
                 *, 
                 gamma: float = 0.99, 
                 lr: float = 3e-4, 
                 alpha_lr: float = 3e-2,
                 tau: float = 0.005, 
                 lr_schedule_end=5e-5,
                 delay_update: int = 2,
                 use_target_feature: bool = True,
                 reward_loss_wgt: int = 0.5):
        
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.use_target_feature = use_target_feature
        self.reward_loss_wgt = reward_loss_wgt
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.feature_optim = optax.adam(lr)

        self.state = QSMRepTrainState(
            params=params,
            opt_state=QSMRepOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                q_score=self.policy_optim.init(params.q_score),
                log_alpha=self.alpha_optim.init(jnp.array(params.log_alpha)),
                feature=self.feature_optim.init(
                    {'feature': params.feature, 'mu': params.mu, 'theta': params.theta}),
            ),
            step=jnp.int32(0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: QSMRepTrainState, data: Experience
        ) -> Tuple[QSMRepTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            step = state.step
            (q1_params, q2_params, target_q1_params, target_q2_params, q_score_params, 
             log_alpha, feat_params, target_feat_params, mu_params, theta_params) = state.params
            q1_opt_state, q2_opt_state, q_score_opt_state, log_alpha_opt_state, feature_opt_state = state.opt_state
            next_action_key = key

            # --------- feature step ----------
            def feature_loss_fn(params: hk.Params) -> jax.Array:
                feat = self.agent.feature(params['feature'], obs, action)
                mu = self.agent.mu(params['mu'], next_obs)

                contrastive = jnp.sum(
                    feat[:, None, :] * mu[None, :, :], axis=-1)
                ce = -jnp.mean(jnp.diag(jax.nn.log_softmax(contrastive)))
                feature_loss = ce
                r_loss = 0.0
                if self.reward_loss_wgt > 0:
                    r_hat = self.agent.theta(params['theta'], feat)
                    r_loss = jnp.mean((r_hat - reward) ** 2)
                feature_loss += self.reward_loss_wgt * r_loss
                return feature_loss, (r_loss, ce)

            feat_step_params = {'feature': feat_params,
                                'mu': mu_params, 'theta': theta_params}
            (feat_loss, (r_loss, ce)), feat_grads = jax.value_and_grad(
                feature_loss_fn, has_aux=True)(feat_step_params)
            feature_update, feature_opt_state = self.feature_optim.update(
                feat_grads, feature_opt_state)
            feat_step_params = optax.apply_updates(
                feat_step_params, feature_update)

            feat_params = feat_step_params['feature']
            mu_params = feat_step_params['mu']
            theta_params = feat_step_params['theta']
            # update target feature
            if self.use_target_feature:
                target_feat_params = optax.incremental_update(feat_params, target_feat_params, tau)

            # ----- rest is standard QSM with feature-based critic ------
            # compute target q
            act_feat_params = target_feat_params if self.use_target_feature else feat_params
            params = (q_score_params, log_alpha, q1_params, q2_params, act_feat_params, mu_params, theta_params)
            next_action = self.agent.get_action(
                next_action_key, params, next_obs)

            feat = self.agent.feature(act_feat_params, obs, action)
            next_feat = self.agent.feature(act_feat_params, next_obs, next_action)

            q1_target = self.agent.q(target_q1_params, next_feat)
            q2_target = self.agent.q(target_q2_params, next_feat)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # update q
            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, feat)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update q_score
            def q_score_loss_fn(q_score_params: hk.Params) -> jax.Array:
                q1, q1_score = self.agent.get_q_score_from_gradient(
                    act_feat_params, q1_params, obs, action)
                q2, q2_score = self.agent.get_q_score_from_gradient(
                    act_feat_params, q2_params, obs, action)
                q_score = self.agent.q_score(q_score_params, obs, action)
                q_minimum_score = jnp.where(
                    q1.reshape(-1, 1) < q2.reshape(-1, 1), q1_score, q2_score)
                q_score_loss = jnp.mean((q_score - q_minimum_score) ** 2)
                return q_score_loss, (q1, q2)

            (q_score_loss, aux), q_score_grads = jax.value_and_grad(
                q_score_loss_fn, has_aux=True)(q_score_params)
            q1, q2 = aux
            q_score_update, q_score_opt_state = self.policy_optim.update(
                q_score_grads, q_score_opt_state)
            q_score_params = optax.apply_updates(
                q_score_params, q_score_update)
            
            # update target q
            target_q1_params = optax.incremental_update(
                q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(
                q2_params, target_q2_params, self.tau)

            # update log_alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * \
                    jnp.log(2 * jnp.pi * jnp.exp(1) *
                            (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * \
                    (-1 * jax.lax.stop_gradient(approx_entropy) +
                     self.agent.target_entropy)
                return log_alpha_loss
            
            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            update, log_alpha_opt_state = self.alpha_optim.update(log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, update)

            state = QSMRepTrainState(
                params=QSMRepParams(
                    q1_params, q2_params, target_q1_params, target_q2_params, q_score_params, log_alpha,
                    feat_params, target_feat_params, mu_params, theta_params),
                opt_state=QSMRepOptStates(
                    q1_opt_state, q2_opt_state, q_score_opt_state, log_alpha_opt_state, feature_opt_state),
                step=step + 1
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "q_score_loss": q_score_loss,
                'r_loss': r_loss,
                'feature_ce_loss': ce,
                'total_feature_loss': feat_loss,
                "alpha": jnp.exp(log_alpha),
            }
            return state, info

        self._implement_common_behavior(
            stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        feat_params = self.state.params.target_feature if self.use_target_feature else self.state.params.feature
        return (self.state.params.q_score, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, 
                feat_params, self.state.params.mu, self.state.params.theta)
