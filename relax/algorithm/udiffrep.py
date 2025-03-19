from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.sdac import Diffv2Params
from relax.algorithm.sdac import Diffv2TrainState, Diffv2OptStates
from relax.network.udiffrep import UDiffRepNet, UDiffRepParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class UDiffRepOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    feature: optax.OptState


class UDiffRepTrainState(NamedTuple):
    params: UDiffRepParams
    opt_state: UDiffRepOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class UDiffRep(Algorithm):

    def __init__(
        self,
        agent: UDiffRepNet,
        params: UDiffRepParams,
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
        use_target_feature: bool = True,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.use_target_feature = use_target_feature
        self.optim = optax.adam(lr)
        self.feature_optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = UDiffRepTrainState(
            params=params,
            opt_state=UDiffRepOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                feature=self.feature_optim.init({'feature': params.feature, 'mu': params.mu}),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            (q1_params, q2_params, target_q1_params, target_q2_params,
              policy_params, target_policy_params, log_alpha,
                feat_params, target_feat_params, mu_params) = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, feature_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, feat_loss_key, policy_loss_key = jax.random.split(key, 3)
            reward *= self.reward_scale

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(
                        params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            # --------- feature step ----------
            def feature_loss_fn(params: hk.Params) -> jax.Array:
                t_key, noise_key = jax.random.split(feat_loss_key)
                t = jax.random.randint(t_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
                noise = jax.random.normal(noise_key, action.shape)
                act_noisy = jax.vmap(self.agent.diffusion.q_sample)(t, action, noise)

                mu = self.agent.mu(params['mu'], next_obs)
                def _feat_loss(_act):
                    feat = self.agent.feature(params['feature'], obs, _act, t)
                    return jnp.sum(feat[..., None, :] @ mu[..., :, None])
                noise_pred = jax.grad(_feat_loss)(act_noisy)
                feature_loss = jnp.mean((noise_pred - noise) ** 2)
                return feature_loss

            feat_step_params = {'feature': feat_params, 'mu': mu_params}
            feat_loss, feat_grads = jax.value_and_grad(feature_loss_fn, has_aux=False)(feat_step_params)
            feature_update, feature_opt_state = self.feature_optim.update(feat_grads, feature_opt_state)
            feat_step_params = optax.apply_updates(feat_step_params, feature_update)
            
            feat_params = feat_step_params['feature']
            mu_params = feat_step_params['mu']
            target_feat_params = optax.incremental_update(feat_params, target_feat_params, tau)

            # ----- rest is standard SDAC with feature-based critic ------

            # ----- critic step -----
            def get_min_q(feat):
                q1 = self.agent.q(q1_params, feat)
                q2 = self.agent.q(q2_params, feat)
                q = jnp.minimum(q1, q2)
                return q
            
            act_feat_params = target_feat_params if self.use_target_feature else feat_params
            params = (policy_params, log_alpha, q1_params, q2_params, act_feat_params, mu_params) 
            # NOTE: We are setting t=0 for obtaining current feature, next feature, this might not be the best way!
            feat = self.agent.feature(act_feat_params, obs, action, jnp.zeros(obs.shape[0]))
            next_action = self.agent.get_action(next_eval_key, params, next_obs)
            next_feat = self.agent.feature(act_feat_params, next_obs, next_action, jnp.zeros(obs.shape[0]))

            q1_target = self.agent.q(target_q1_params, next_feat)
            q2_target = self.agent.q(target_q2_params, next_feat)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, feat)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # ---- policy step -------
            def policy_loss_fn(policy_params) -> jax.Array:
                q_min = get_min_q(next_feat)
                q_mean, q_std = q_min.mean(), q_min.std()
                norm_q = q_min - running_mean / running_std
                scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                q_weights = jnp.exp(scaled_q)
                # q_weights = q_weights
                def denoiser(t, x):
                    d_phi_d_a = jax.vmap(jax.jacfwd(
                        lambda _x, _obs, _t:  self.agent.feature(act_feat_params, _obs, _x, _t)))(x, next_obs, t)
                    return self.agent.policy(policy_params, d_phi_d_a)
                t_key, noise_key = jax.random.split(policy_loss_key)
                t = jax.random.randint(t_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
                _, _, loss = self.agent.diffusion.weighted_p_loss(noise_key, q_weights, denoiser, t,
                                                            jax.lax.stop_gradient(next_action))

                return loss, (q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # ---- misc -------
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

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = UDiffRepTrainState(
                params=UDiffRepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params,
                                     log_alpha, feat_params, target_feat_params, mu_params),
                opt_state=UDiffRepOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, 
                                           log_alpha=log_alpha_opt_state, feature=feature_opt_state),
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
                'feature_loss': feat_loss,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        feat_params = self.state.params.target_feature if self.use_target_feature else self.state.params.feature
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, feat_params, self.state.params.mu)

    def get_policy_params_to_save(self):
        return (self.state.params.target_policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.feature, self.state.params.mu)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
