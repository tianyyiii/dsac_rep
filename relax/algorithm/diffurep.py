from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.diffurep import DiffURepNet, DiffURepParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class DiffURepOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    phi: optax.OptState
    mu: optax.OptState
    log_alpha: optax.OptState


class DiffURepTrainState(NamedTuple):
    params: DiffURepParams
    opt_state: DiffURepOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class DiffURep(Algorithm):

    def __init__(
        self,
        agent: DiffURepNet,
        params: DiffURepParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
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
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        lr_schedule_mu = optax.schedules.linear_schedule(
            init_value=lr / 100,
            end_value=lr_schedule_end / 100,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.q_optim = optax.adam(lr)
        self.phi_optim = optax.adam(learning_rate=lr_schedule)
        self.mu_optim = optax.adam(learning_rate=lr_schedule_mu)
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = DiffURepTrainState(
            params=params,
            opt_state=DiffURepOptStates(
                q1=self.q_optim.init(params.q1),
                q2=self.q_optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                phi=self.phi_optim.init(params.phi),
                mu=self.mu_optim.init(params.mu),
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
            key: jax.Array, state: DiffURepTrainState, data: Experience
        ) -> Tuple[DiffURepOptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, phi_params, mu_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, phi_opt_state, mu_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 7)

            reward *= self.reward_scale

            '''
            Critic and feature learning stage
            '''

            def get_min_q(feature):
                q1 = self.agent.q(q1_params, feature)
                q2 = self.agent.q(q2_params, feature)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params, phi_params), next_obs)
            feature_q = self.agent.phi(phi_params, next_obs, next_action, 0)
            q1_target = self.agent.q(target_q1_params, feature_q)
            q2_target = self.agent.q(target_q2_params, feature_q)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = jax.lax.stop_gradient(reward + (1 - done) * self.gamma * q_target)

            def q_loss_fn(q_params: hk.Params, phi_params: hk.Params) -> jax.Array:
                feature_ql = self.agent.phi(phi_params, obs, action, 0)
                q = self.agent.q(q_params, feature_ql)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), (q1_grads, phi_grads_q1) = jax.value_and_grad(q_loss_fn, argnums=(0, 1), has_aux=True)(q1_params, phi_params)
            (q2_loss, q2), (q2_grads, phi_grads_q2) = jax.value_and_grad(q_loss_fn, argnums=(0, 1), has_aux=True)(q2_params, phi_params)
            # q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            # q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            # q1_params = optax.apply_updates(q1_params, q1_update)
            # q2_params = optax.apply_updates(q2_params, q2_update)


            def get_feature_a(phi_params, x, obs, t):
                def single_sample_jacobian(x_sample, obs_sample, t_sample):
                    return jax.jacfwd(lambda x: self.agent.phi(phi_params, obs_sample, x, t_sample))(x_sample)
                if len(obs.shape) == 2:
                    if t.ndim == 0:
                        t = jnp.full((obs.shape[0],), t)
                    feature_a = jax.vmap(single_sample_jacobian)(x, obs, t)
                    feature_a1 = jnp.reshape(feature_a, (obs.shape[0], -1)) 
                else:
                    feature_a = jax.jacfwd(lambda x: self.agent.phi(phi_params, obs, x, t))(x)
                    feature_a1 = jnp.reshape(feature_a, (-1,))
                return feature_a, feature_a1

            '''
            Actor and feature learning stage
            '''


            def policy_loss_fn(policy_params, phi_params, mu_params) -> jax.Array:
                feature_pl = self.agent.phi(phi_params, next_obs, next_action, 0)
                q_min = get_min_q(feature_pl)
                q_mean, q_std = q_min.mean(), q_min.std()
                norm_q = q_min - running_mean / running_std
                scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                q_weights = jnp.exp(scaled_q)
                q_weights = jax.lax.stop_gradient(q_weights)

                def denoiser(t, x):
                    _, feature_a = get_feature_a(phi_params=phi_params, x=x, obs=next_obs, t=t)
                    return self.agent.policy(policy_params, feature_a)
                
                t = jax.random.randint(diffusion_time_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
                noise, x_noisy, loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, 
                                                                            q_weights, 
                                                                            denoiser, t, 
                                                                            jax.lax.stop_gradient(next_action))

                phi_output, _ = get_feature_a(phi_params, x_noisy, obs, t)
                phi_output = jnp.transpose(phi_output, (0, 2, 1))
                if self.rep_weight > 0.0:
                    mu_output = self.agent.mu(mu_params, next_obs)
                    mul = jnp.matmul(phi_output, mu_output[..., None])
                    mul = mul.squeeze(-1)
                    rep_loss = optax.squared_error(mul, noise).mean()
                    loss += self.rep_weight * rep_loss

                return loss, (rep_loss, q_weights, scaled_q, q_mean, q_std)

            (policy_loss, (rep_loss, q_weights, scaled_q, q_mean, q_std)), (policy_grads, phi_grads_p, mu_grads) = jax.value_and_grad(policy_loss_fn, argnums=(0, 1, 2), has_aux=True)(policy_params, phi_params, mu_params)

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

            phi_grads_q = jax.tree_util.tree_map(lambda a, b: a + b, phi_grads_q1, phi_grads_q2)
            phi_grads = jax.tree_util.tree_map(lambda a, c: a + c, phi_grads_q, phi_grads_p)

            q1_params, q1_opt_state = param_update(self.q_optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.q_optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            phi_params, phi_opt_state = param_update(self.phi_optim, phi_params, phi_grads, phi_opt_state)
            mu_params, mu_opt_state = param_update(self.mu_optim, mu_params, mu_grads, mu_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = DiffURepTrainState(
                params=DiffURepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, phi_params, mu_params, log_alpha),
                opt_state=DiffURepOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, phi=phi_opt_state, mu=mu_opt_state, log_alpha=log_alpha_opt_state),
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
                "policy_loss": policy_loss,
                "rep_loss": rep_loss,
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
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.phi)

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.phi)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
