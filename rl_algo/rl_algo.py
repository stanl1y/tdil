from .continuous import *
from .discrete import *


def get_rl_agent(env, config):
    if config.algo == "sac":
        return sac(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
            log_alpha_init=config.log_alpha_init,
            no_update_alpha=config.no_update_alpha,
        )
    elif config.algo == "ddpg":
        return ddpg(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
            policy_type=config.policy_type,
        )
    elif config.algo == "td3":
        return td3(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            tau=config.tau,
            batch_size=config.batch_size,
            use_ounoise=config.use_ounoise,
            policy_update_delay=config.policy_update_delay
        )
    elif config.algo == "rainbow_dqn":
        return rainbow_dqn(
            observation_dim=env.get_observation_dim(),
            action_num=env.get_action_dim(),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            optim=config.optim,
            network_type=config.network_type,
            noisy_network=config.noisy_network,
            double_dqn=config.double_dqn,
            soft_update_target=config.soft_update_target,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            lr=config.lr,
            tau=config.tau,
            batch_size=config.batch_size,
        )
    elif config.algo == "discrete_sac":
        return discrete_sac(
            observation_dim=env.get_observation_dim(),
            action_num=env.get_action_dim(),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            optim=config.optim,
            soft_update_target=config.soft_update_target,
            lr=config.lr,
            tau=config.tau,
            batch_size=config.batch_size,
        )
    elif config.algo == "ppo":
        return ppo(
            observation_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim(),
            action_lower=min(env.action_space.low),
            action_upper=max(env.action_space.high),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            lambda_decay=config.lambda_decay,
            actor_optim=config.actor_optim,
            critic_optim=config.critic_optim,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            target_kl=config.target_kl,
            ppo_clip_value=config.ppo_clip_value,
            batch_size=config.batch_size,
            max_policy_train_iters=config.max_policy_train_iters,
            value_train_iters=config.value_train_iters,
            para_std=config.para_std,
            action_std = config.action_std,
            action_std_decay_rate = config.action_std_decay_rate,
            min_action_std = config.min_action_std,
            action_std_decay_freq = config.action_std_decay_freq,
        )
    else:
        raise TypeError(f"rl agent type : {config.algo} not supported")
