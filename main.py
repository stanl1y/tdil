import argparse
import yaml
from rl_algo import get_rl_agent, get_util
from environment import get_env
from replay_buffer import get_replay_buffer
from main_stage import get_main_stage


def get_config():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--main_task",
        type=str,
        default="sac",
        help="which RL algo",
    )
    given_configs, remaining = parser.parse_known_args()
    with open(f"config_files/{given_configs.main_task}.yml", "r") as f:
        hyper = yaml.safe_load(f)
        parser.set_defaults(**hyper)

    parser.add_argument("--buffer_size", type=int, help="size of replay buffer")
    parser.add_argument(
        "--wrapper_type",
        type=str,
        help='basic for normal environment, gym robotic for training on robotic envs "directly", her is for training on robotic envs with hindsight experience replay',
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Gym environment name, default: CartPole-v0",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes, default: 100")
    parser.add_argument("--total_timesteps", type=int, help="total timesteps, default: 2000")
    parser.add_argument("--seed", type=int, help="Seed, default: 1")
    parser.add_argument(
        "--main_stage_type",
        type=str,
        help="type of main stage(ex. off_policy, her_off_policy, collect_expert)",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size, default: 256")
    parser.add_argument("--hidden_dim", type=int, help="dimension of hidden layer 256")
    parser.add_argument(
        "--use_ounoise", action="store_true", help="use ou noise or not"
    )
    parser.add_argument(
        "--continue_training", action="store_true", help="use ou noise or not"
    )
    parser.add_argument(
        "--ood", action="store_true", help="Out of distribution evaluation"
    )
    parser.add_argument(
        "--perturb_from_mid",
        action="store_true",
        default=False,
        help="When eval in ood mode, perturb from mid",
    )
    parser.add_argument(
        "--perturb_with_repeated_action",
        action="store_true",
        default=False,
        help="Perturb agent with repeated action instead of random action",
    )
    parser.add_argument(
        "--perturb_step_num", type=int, default=10, help="number of perturb step"
    )
    parser.add_argument(
        "--expert_transition_num", type=int, help="number of expert data"
    )
    parser.add_argument(
        "--expert_sub_sample_ratio",
        type=float,
        help="propotion of expert data left in training(-1 means no subsample)",
    )
    parser.add_argument(
        "--policy_threshold_ratio",
        type=float,
        help="threshold ratio for filtering out expert-like data in policy training",
    )
    parser.add_argument(
        "--save_env_states",
        action="store_true",
        help="store the env state for 'set_state' or not",
    )
    parser.add_argument("--expert_episode_num", type=int, help="number of expert data")
    parser.add_argument(
        "--buffer_warmup_step",
        type=int,
        help="number of step of random walk in the initial of training",
    )
    parser.add_argument(
        "--render", action="store_true", help="whether to render when testing"
    )
    parser.add_argument(
        "--noisy_network", action="store_true", help="whether to render when testing"
    )
    parser.add_argument("--infinite_neighbor_buffer", action="store_true")
    parser.add_argument(
        "--no_bc",
        action="store_true",
        help="don't use behavior cloning when neighborhood il",
    )
    parser.add_argument(
        "--use_relative_reward",
        action="store_true",
        help="use relative reward when training neighborhood il",
    )
    parser.add_argument(
        "--no_hard_negative_sampling",
        action="store_true",
        help="don't use hard negative sampling in neighborhood il",
    )
    parser.add_argument(
        "--bc_pretraining",
        action="store_true",
        help="use behavioral cloning to pretrain policy",
    )
    parser.add_argument(
        "--easy_nagative_weight_decay",
        action="store_true",
        help="decay the weight of easy negative samples",
    )
    parser.add_argument(
        "--no_update_alpha", action="store_true", help="don't update sac's alpha"
    )
    parser.add_argument(
        "--terminate_when_unhealthy",
        action="store_true",
        help="terminate when unhealthy",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="set state in neighborhood il",
    )
    parser.add_argument(
        "--state_only",
        action="store_true",
        help="neighborhood il without using expert action",
    )
    parser.add_argument(
        "--critic_without_entropy",
        action="store_true",
        help="exclude entropy term in target value of critic",
    )
    parser.add_argument(
        "--use_true_expert_relative_reward",
        action="store_true",
        help="use true expert relative reward or pure ones",
    )
    parser.add_argument(
        "--low_hard_negative_weight",
        action="store_true",
        help="weight of hard negative times (1-alpha)",
    )
    parser.add_argument(
        "--use_top_k",
        action="store_true",
        help="usetop k reward in neighborhood il",
    )
    parser.add_argument(
        "--k_of_topk",
        type=int,
        help="value of k",
    )
    parser.add_argument(
        "--use_IDM",
        action="store_true",
        help="use InverseDynamicModule in neighborhood il (only for state only)",
    )
    parser.add_argument(
        "--use_pretrained_neighbor",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_neighbor_weight_path",
        type=str,
    )
    parser.add_argument(
        "--target_entropy_weight",
        type=float,
        help="target entropy weight for sac",
    )
    parser.add_argument(
        "--neighbor_model_alpha",
        type=float,
    )
    parser.add_argument(
        "--reward_scaling_weight", type=float, help="scale of relative reward"
    )
    parser.add_argument(
        "--entropy_loss_weight_decay_rate",
        type=float,
        help="decay rate of entropy loss weight",
    )
    parser.add_argument(
        "--log_alpha_init", type=float, help="initial value of log_alpha"
    )
    parser.add_argument(
        "--tau", type=float, help="tau for soft update of target network"
    )
    parser.add_argument(
        "--neighborhood_tau",
        type=float,
        help="tau for soft update of target neighborhood network",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        help="nane of log file",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="",
        help="name of expert data",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        help="path of trained model weight",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        help="what kind of replay buffer",
    )
    parser.add_argument(
        "--explore_step", type=int, help="number of step of exploration in set_state_il"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="max step of each episode"
    )
    parser.add_argument(
        "--infinite_bootstrap",
        action="store_true",
        help="infinite bootstrapping (done is always false)",
    )
    parser.add_argument(
        "--reset_as_expert_state",
        action="store_true",
        help="reset env as first state of expert state",
    )
    parser.add_argument(
        "--bc_only",
        action="store_true",
        help="only train policy with bc loss",
    )
    parser.add_argument(
        "--initial_state_noise_std",
        type=float,
        help="std of noise added to initial state",
    )
    parser.add_argument(
        "--complementary_reward",
        action="store_true",
        help="use complementary reward in neighborhood il",
    )
    parser.add_argument(
        "--only_use_relative_state",
        action="store_true",
        help="delete absolute state dimension from env observation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="discount factor of td learning",
    )
    parser.add_argument(
        "--toy_reward_type",
        type=str,
        help="reward type of toy env",
    )
    parser.add_argument(
        "--use_discriminator",
        action="store_true",
        help="use irl reward in neighborhood il",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="hyper balancing irl reward and neighborhood reward",
    )
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


if __name__ == "__main__":
    config, config_text = get_config()
    env = get_env(
        config.env,
        config.wrapper_type,
        config.terminate_when_unhealthy,
        config.max_episode_steps,
        config.only_use_relative_state,
        config.toy_reward_type,
    )
    agent = get_rl_agent(env, config)
    storage = get_replay_buffer(env, config)
    main_fn = get_main_stage(config)
    if hasattr(config, "util"):
        util_dict = get_util(env, config)
        main_fn.start(agent, env, storage, util_dict)
    else:
        main_fn.start(agent, env, storage)
