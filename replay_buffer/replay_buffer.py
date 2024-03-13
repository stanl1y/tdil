from .normal_replay_buffer import normal_replay_buffer
from .onpolicy_replay_buffer import onpolicy_replay_buffer
from .her_replay_buffer import her_replay_buffer

def get_replay_buffer(env, config):
    if config.buffer_type == "normal":
        return normal_replay_buffer(
            size=config.buffer_size,
            state_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim() if env.is_continuous() else 1,
            save_env_states=config.save_env_states,
            save_state_idx=config.save_state_idx,
            to_tensor=config.buffer_to_tensor,
        )
    elif config.buffer_type == "onpolicy":
        return onpolicy_replay_buffer(
            size=config.buffer_size,
            state_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim() if env.is_continuous() else 1,
        )
    elif config.buffer_type == "her":
        return her_replay_buffer(
            size=config.buffer_size,
            state_dim=env.get_state_dim(),
            goal_dim=env.get_goal_dim(),
            action_dim=env.get_action_dim() if env.is_continuous() else 1,
            reward_fn=env.compute_reward
        )
    else:
        raise TypeError(f"replay buffer type : {config.buffer_type} not supported")
