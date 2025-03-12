from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.qsm import QSMNet, QSMParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class QSMOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    q_score: optax.OptState
    log_alpha: optax.OptState

class QSMTrainState(NamedTuple):
    params: QSMParams
    opt_state: QSMOptStates


class QSM(Algorithm):
    def __init__(self, agent: QSMNet, params: QSMParams, *, gamma: float = 0.99, lr: float = 3e-4, 
                 alpha_lr: float = 3e-2, tau: float = 0.005, lr_schedule_end=5e-5):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)

        self.state = QSMTrainState(
            params=params,
            opt_state=QSMOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                q_score=self.policy_optim.init(params.q_score),
                log_alpha=self.alpha_optim.init(jnp.array(params.log_alpha)),
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: QSMTrainState, data: Experience
        ) -> Tuple[QSMTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, q_score_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, q_score_opt_state, log_alpha_opt_state = state.opt_state
            next_action_key = key

            # compute target q
            next_action = self.agent.get_action(next_action_key, (q_score_params, log_alpha, q1_params, q2_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # update q
            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
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
                q1, q1_score = self.agent.get_q_score_from_gradient(q1_params, obs, action)
                q2, q2_score = self.agent.get_q_score_from_gradient(q2_params, obs, action)
                q_score = self.agent.q_score(q_score_params, obs, action)
                q_minimum_score = jnp.where(q1.reshape(-1, 1) < q2.reshape(-1, 1), q1_score, q2_score)
                q_score_loss = jnp.mean((q_score - q_minimum_score) ** 2)
                return q_score_loss, (q1, q2)

            (q_score_loss, aux), q_score_grads = jax.value_and_grad(q_score_loss_fn, has_aux=True)(q_score_params)
            q1, q2 = aux
            q_score_update, q_score_opt_state = self.policy_optim.update(q_score_grads, q_score_opt_state)
            q_score_params = optax.apply_updates(q_score_params, q_score_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

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
            update, log_alpha_opt_state = self.alpha_optim.update(
                log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, update)

            state = QSMTrainState(
                params=QSMParams(q1_params, q2_params, target_q1_params,
                                 target_q2_params, q_score_params, log_alpha),
                opt_state=QSMOptStates(q1_opt_state, q2_opt_state, q_score_opt_state, log_alpha_opt_state),
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "q_score_loss": q_score_loss,
                "alpha": jnp.exp(log_alpha),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return self.state.params.q_score, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2
