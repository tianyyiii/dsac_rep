from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffrep_image import DiffRepImageNet, DiffRepImageParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class DiffRepImageOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    mu: optax.OptState
    encoder_v: optax.OptState
    log_alpha: optax.OptState


class DiffRepImageTrainState(NamedTuple):
    params: DiffRepImageParams
    opt_state: DiffRepImageOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class DiffRepImage(Algorithm):

    def __init__(
        self,
        agent: DiffRepImageNet,
        params: DiffRepImageParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        encoder_lr: float = 1e-4,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        rep_weight: float = 1.0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.encoder_optim = optax.adam(encoder_lr)
        self.entropy = 0.0

        self.state = DiffRepImageTrainState(
            params=params,
            opt_state=DiffRepImageOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                mu=self.policy_optim.init(params.mu),
                encoder_v=self.encoder_optim.init(params.encoder_v),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.rep_weight = rep_weight

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DiffRepImageTrainState, data: Experience
        ) -> Tuple[DiffRepImageOptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, mu_params, encoder_v_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, mu_opt_state, encoder_v_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 7)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            def get_min_target_q(s, a):
                q1 = self.agent.q(target_q1_params, s, a)
                q2 = self.agent.q(target_q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params, encoder_v_params: hk.Params) -> Tuple[jax.Array, jax.Array]:
                encoded_obs = self.agent.encoder_v(encoder_v_params, obs)
                q = self.agent.q(q_params, encoded_obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), (q1_grads, encoder_grad_q1) = jax.value_and_grad(q_loss_fn, argnums=(0, 1), has_aux=True)(q1_params, encoder_v_params)
            (q2_loss, q2), (q2_grads, encoder_grad_q2) = jax.value_and_grad(q_loss_fn, argnums=(0, 1), has_aux=True)(q2_params, encoder_v_params)

            # q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            # q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            # q1_params = optax.apply_updates(q1_params, q1_update)
            # q2_params = optax.apply_updates(q2_params, q2_update)


            def policy_loss_fn(policy_params, mu_params, encoder_v_params: hk.Params) -> jax.Array:
                encoded_p_obs = self.agent.encoder_v(encoder_v_params, obs)
                encoded_p_next_obs = self.agent.encoder_v(encoder_v_params, next_obs)
                q_min = get_min_q(encoded_p_next_obs, next_action)
                q_mean, q_std = q_min.mean(), q_min.std()
                norm_q = q_min - running_mean / running_std
                scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                q_weights = jnp.exp(scaled_q)
                def denoiser(t, x):
                    return self.agent.policy(policy_params, encoded_p_next_obs, x, t)[1]
                t = jax.random.randint(diffusion_time_key, (encoded_p_next_obs.shape[0],), 0, self.agent.num_timesteps)
                noise, x_noisy, loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, q_weights, denoiser, t,
                                                            jax.lax.stop_gradient(next_action))
                
                phi_output = self.agent.policy(policy_params, encoded_p_obs, x_noisy, t)[0]
                mu_output = self.agent.mu(mu_params, encoded_p_next_obs)
                mul = jnp.matmul(phi_output, mu_output[..., None])
                mul = mul.squeeze(-1)
                rep_loss = optax.squared_error(mul, noise).mean()
                loss = loss + self.rep_weight * rep_loss

                return loss, (q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), (policy_grads, mu_grads, encoder_grad_policy) = jax.value_and_grad(policy_loss_fn, argnums=(0, 1, 2), has_aux=True)(policy_params, mu_params, encoder_v_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            mu_params, mu_opt_state = delay_param_update(self.policy_optim, mu_params, mu_grads, mu_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            total_encoder_grad = encoder_grad_q1 + encoder_grad_policy
            encoder_v_params, encoder_v_opt_state = param_update(self.encoder_optim, encoder_v_params, total_encoder_grad, encoder_v_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = DiffRepImageTrainState(
                params=DiffRepImageParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, mu_params, log_alpha),
                opt_state=DiffRepImageOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, mu=mu_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
