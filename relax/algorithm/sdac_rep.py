from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.sdac import Diffv2Params
from relax.algorithm.sdac import Diffv2TrainState, Diffv2OptStates
from relax.network.sdac_rep import SDACRepNet, SDACRepParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class SDACRepOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    feature: optax.OptState


class SDACRepTrainState(NamedTuple):
    params: SDACRepParams
    opt_state: SDACRepOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class SDACRep(Algorithm):

    def __init__(
        self,
        agent: SDACRepNet,
        params: SDACRepParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_feat: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        use_target_feature: bool = True,
        reward_loss_wgt: float = 0.5,
        extra_feature_steps: int = 0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.use_target_feature = use_target_feature
        self.reward_loss_wgt = reward_loss_wgt
        self.extra_feature_steps = extra_feature_steps
        self.optim = optax.adam(lr)
        self.feature_optim = optax.adam(lr_feat)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = SDACRepTrainState(
            params=params,
            opt_state=SDACRepOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                feature=self.feature_optim.init({'feature': params.feature,
                                                 'mu': params.mu,
                                                 'theta': params.theta}),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema

        def delay_target_update(step, params, target_params, tau):
            return jax.lax.cond(
                step % self.delay_update == 0,
                lambda target_params: optax.incremental_update(
                    params, target_params, tau),
                lambda target_params: target_params,
                target_params
            )

        def feature_step(state, data):
            def feature_loss_fn(params: hk.Params) -> jax.Array:
                feat = self.agent.feature(params['feature'], obs, action)
                mu = self.agent.mu(params['mu'], next_obs)

                contrastive = jnp.sum(
                    feat[:, None, :] * mu[None, :, :], axis=-1)
                ce = -jnp.mean(jnp.diag(jax.nn.log_softmax(contrastive)))
                r_loss = 0.0
                if self.reward_loss_wgt > 0:
                    rhat = self.agent.theta(params['theta'], feat)
                    r_loss = jnp.mean((rhat - reward) ** 2)
                feature_loss = ce + self.reward_loss_wgt * r_loss
                return feature_loss, (ce, r_loss)
            
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            (q1_params, q2_params, target_q1_params, target_q2_params,
             policy_params, target_policy_params, log_alpha,
                feat_params, target_feat_params, mu_params, theta_params) = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, feature_opt_state = state.opt_state
            step = state.step

            feat_step_params = {'feature': feat_params,
                                'mu': mu_params, 'theta': theta_params}
            (feat_loss, (ce, r_loss)), feat_grads = jax.value_and_grad(
                feature_loss_fn, has_aux=True)(feat_step_params)
            feature_update, feature_opt_state = self.feature_optim.update(
                feat_grads, feature_opt_state)
            feat_step_params = optax.apply_updates(
                feat_step_params, feature_update)

            feature_params = feat_step_params['feature']
            mu_params = feat_step_params['mu']
            theta_params = feat_step_params['theta']
            target_feat_params = delay_target_update(
                step, feature_params, target_feat_params, self.tau)
            feat_metrics = {'r_loss': r_loss,
                            'feature_ce_loss': ce,
                            'total_feature_loss': feat_loss}
            return (feature_params, mu_params, theta_params, target_feat_params), feature_opt_state, feat_metrics

        @jax.jit
        def stateless_update(
            key: jax.Array, state: SDACRepTrainState, data: Experience
        ) -> Tuple[SDACRepOptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            (q1_params, q2_params, target_q1_params, target_q2_params,
              policy_params, target_policy_params, log_alpha,
                feat_params, target_feat_params, mu_params, theta_params) = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, feature_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(key, 3)
            reward *= self.reward_scale

            # --------- feature step ----------
            ((feature_params, mu_params, theta_params, target_feat_params), \
                feature_opt_state, feat_metrics) = feature_step(state, data)

            # ----- rest is standard SDAC with feature-based critic ------
            def get_min_q(feat):
                q1 = self.agent.q(q1_params, feat)
                q2 = self.agent.q(q2_params, feat)
                q = jnp.minimum(q1, q2)
                return q
            
            act_feat_params = target_feat_params if self.use_target_feature else feature_params
            params = (policy_params, log_alpha, q1_params, q2_params, act_feat_params, mu_params, theta_params) 
            next_action = self.agent.get_action(next_eval_key, params, next_obs)
            if self.use_target_feature:
                feat = self.agent.feature(target_feat_params, obs, action)
                next_feat = self.agent.feature(target_feat_params, next_obs, next_action)
            else:
                feat = self.agent.feature(feat_params, obs, action)
                next_feat = self.agent.feature(feat_params, obs, action)

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


            def policy_loss_fn(policy_params) -> jax.Array:
                q_min = get_min_q(next_feat)
                q_mean, q_std = q_min.mean(), q_min.std()
                norm_q = q_min - running_mean / running_std
                scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                q_weights = jnp.exp(scaled_q)
                # q_weights = q_weights
                def denoiser(t, x):
                    return self.agent.policy(policy_params, next_obs, x, t)
                t = jax.random.randint(diffusion_time_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
                _, _, loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, q_weights, denoiser, t,
                                                            jax.lax.stop_gradient(next_action))

                return loss, (q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

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

            target_q1_params = delay_target_update(step, q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(step, q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(step, policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = SDACRepTrainState(
                params=SDACRepParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params,
                                     log_alpha, feature_params, target_feat_params, mu_params, theta_params),
                opt_state=SDACRepOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, 
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
                **feat_metrics
            }
            return state, info
        
        @jax.jit
        def update_aux(
            key: jax.Array, state: SDACRepTrainState, data: Experience
        ) -> Tuple[SDACRepOptStates, Metric]:
            ((feature_params, mu_params, theta_params, target_feat_params),
                feature_opt_state, feat_metrics) = feature_step(state, data)
            state = SDACRepTrainState(
                params=SDACRepParams(state.params.q1, state.params.q2, 
                                     state.params.target_q1, state.params.target_q2, 
                                     state.params.policy, state.params.target_policy,
                                     state.params.log_alpha, feature_params, target_feat_params, mu_params, theta_params),
                opt_state=SDACRepOptStates(q1=state.opt_state.q1, q2=state.opt_state.q2, 
                                           policy=state.opt_state.policy,
                                           log_alpha=state.opt_state.log_alpha, feature=feature_opt_state),
                step=state.step ,
                entropy=jnp.float32(0.0),
                running_mean=state.running_mean,
                running_std=state.running_std
            )
            return state, feat_metrics

        self._implement_common_behavior(
            stateless_update, self.agent.get_action, self.agent.get_deterministic_action, update_aux=update_aux)

    def get_policy_params(self):
        feat_params = self.state.params.target_feature if self.use_target_feature else self.state.params.feature
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, feat_params, self.state.params.mu, self.state.params.theta)

    def get_policy_params_to_save(self):
        return (self.state.params.target_policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.feature, self.state.params.mu, self.state.params.theta)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
