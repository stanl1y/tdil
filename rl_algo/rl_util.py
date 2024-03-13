from .neighborhood_il import *


def get_util(env, config):
    util_dict = {}
    for util in config.util:
        if util == "NeighborhoodNet":
            util_dict[util] = NeighborhoodNet(
                input_dim=env.get_observation_dim() * 2, hidden_dim=config.hidden_dim
            )
        if util == "OracleNeighborhoodNet":
            util_dict[util] = OracleNeighborhoodNet(
                input_dim=env.get_observation_dim() * 2, hidden_dim=config.hidden_dim
            )
        if util == "InverseDynamicModule":
            util_dict[util] = InverseDynamicModule(
                input_dim=env.get_observation_dim() * 2,
                hidden_dim=config.hidden_dim,
                output_dim=env.get_action_dim(),
                action_shift=(min(env.action_space.low) + max(env.action_space.high))
                / 2,
                action_scale=(max(env.action_space.high) - min(env.action_space.low))
                / 2,
            )
        if util == "Discriminator":
            util_dict[util] = Discriminator(
                state_dim=env.get_observation_dim(), action_dim=env.get_action_dim(), hidden_dim=config.hidden_dim
            )
    return util_dict
