import gym
from .wrapper import *
from .custom_env import *
import gymnasium


def get_env(
    env_name,
    wrapper_type,
    terminate_when_unhealthy=False,
    max_episode_steps=-1,
    only_use_relative_state=False,
    toy_reward_type='gail'
):
    if env_name in [
        "highway-v0",
        "merge-v0",
        "roundabout-v0",
        "parking-v0",
        "intersection-v0",
        "racetrack-v0",
    ]:
        """
        highway_env:
        https://github.com/eleurent/highway-env
        """
        import highway_env

        env = gym.make(env_name)
    elif env_name in [
        "Maze-v0",
        "Maze-v1",
        "Maze-v2",
        "Maze-v3",
        "Maze-v4",
        "Maze-v5",
        "Maze-v6",
    ]:
        """
        custom_env:
        """
        gym.envs.register(id="Maze-v0", entry_point=Maze_v0, max_episode_steps=250)
        gym.envs.register(id="Maze-v1", entry_point=Maze_v1, max_episode_steps=400)
        gym.envs.register(id="Maze-v2", entry_point=Maze_v2, max_episode_steps=400)
        gym.envs.register(id="Maze-v3", entry_point=Maze_v3, max_episode_steps=400)
        gym.envs.register(id="Maze-v4", entry_point=Maze_v4, max_episode_steps=400)
        gym.envs.register(id="Maze-v5", entry_point=Maze_v5, max_episode_steps=200)
        gym.envs.register(id="Maze-v6", entry_point=Maze_v6, max_episode_steps=200)
        if env_name == "Maze-v2":
            env = gym.make(env_name, **dict(random_reset=True, combine_s_g=True))
        if env_name == "Maze-v6":
            env = gym.make(env_name, **dict(random_reset=True, reward_type=toy_reward_type))
        else:
            env = gym.make(env_name, **dict(random_reset=True))
    elif env_name in [
        "AdroitHandDoor-v1",
        "AdroitHandHammer-v1",
        "AdroitHandPen-v1",
        "AdroitHandRelocate-v1",
    ]:
        if max_episode_steps == -1:
            env = gymnasium.make(env_name)
        else:
            env = gymnasium.make(env_name, max_episode_steps=max_episode_steps)
    else:
        try:
            env = gym.make(env_name, terminate_when_unhealthy=terminate_when_unhealthy)
            print(f"terminate_when_unhealthy:{terminate_when_unhealthy}")
        except:
            env = gym.make(env_name)
    if wrapper_type == "basic":
        return BasicWrapper(env)
    elif wrapper_type == "gym_robotic":
        return GymRoboticWrapper(env)
    elif wrapper_type == "her":
        return HERWrapper(env)
    elif wrapper_type == "normobs":
        return NormObs(env)
    elif wrapper_type == "gymnasium":
        return GymnasiumWrapper(env, only_use_relative_state=only_use_relative_state)
    else:
        raise TypeError(f"env wrapper type : {wrapper_type} not supported")
