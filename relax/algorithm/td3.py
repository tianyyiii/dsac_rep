from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.td3 import TD3Net, TD3Params
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class TD3OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState


class TD3TrainState(NamedTuple):
    params: TD3Params
    opt_state: TD3OptStates
    step: int


class TD3(Algorithm):
    def __init__(self, agent: TD3Net, params: TD3Params, *, gamma: float = 0.99, lr: float = 3e-4,
                 tau: float = 0.005, target_update_freq=1, target_policy_noise=0.2, noise_clip=0.3):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.target_update_freq = target_update_freq
        self.target_policy_noise = target_policy_noise
        self.noise_clip = noise_clip

        self.state = TD3TrainState(
            params=params,
            opt_state=TD3OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
            ),
            step=0
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: TD3TrainState, data: Experience
        ) -> Tuple[TD3TrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params = state.params
            q1_opt_state, q2_opt_state, policy_opt_state = state.opt_state
            step = state.step

            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.target_update_freq == 0,
                    lambda target_params: optax.incremental_update(
                        params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            # ----- CRITIC STEP ----
            # compute target q
            next_action = self.agent.evaluate(target_policy_params, next_obs)
            noise = jax.random.normal(key, next_action.shape) * self.target_policy_noise
            noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
            next_action = jnp.clip(next_action + noise, -1, 1)

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
            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)

            # ----- ACTOR STEP ----
            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                new_action = self.agent.evaluate(policy_params, obs)
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                policy_loss = - jnp.mean(q1 + q2)
                return policy_loss, (q1, q2)

            (policy_loss, (q1, q2)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # ---- update target networks ----
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)
            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

            state = TD3TrainState(
                params=TD3Params(q1_params, q2_params, target_q1_params,
                                 target_q2_params, policy_params, target_policy_params),
                opt_state=TD3OptStates(q1_opt_state, q2_opt_state, policy_opt_state),
                step=step + 1
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
