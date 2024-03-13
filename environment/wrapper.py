import gym
from gym.spaces import Box
from gymnasium.spaces import Box as BoxGymnasium
import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation
from gym import ObservationWrapper


class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_observation_dim(self):
        return self.env.observation_space.shape[0]

    def is_continuous(self):
        return type(self.env.action_space) == Box

    def get_action_dim(self):
        if self.is_continuous():
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n


class GymnasiumWrapper(BasicWrapper):
    def __init__(self, env, only_use_relative_state=False):
        super().__init__(env)
        self.only_use_relative_state = only_use_relative_state

    def is_continuous(self):
        return type(self.env.action_space) == BoxGymnasium

    def get_observation_dim(self):
        if self.only_use_relative_state:
            return self.env.observation_space.shape[0] - 6
        else:
            return self.env.observation_space.shape[0] 

    def reset(self, options=None):
        s = self.env.reset(options=options)[0]
        if self.only_use_relative_state:
            s = np.delete(s, np.s_[29:35])
        return s

    def step(self, action):
        nextState, reward, done, truncated, info = self.env.step(action)
        if self.only_use_relative_state:
            nextState = np.delete(nextState, np.s_[29:35])
        return nextState, reward, truncated, info


class GymRoboticWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = FlattenObservation(
            FilterObservation(env, ["observation", "desired_goal"])
        )


class HERWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_observation_dim(self):
        observation_space = self.env.observation_space
        return (
            observation_space["desired_goal"].shape[0]
            + observation_space["observation"].shape[0]
        )

    def get_state_dim(self):
        observation_space = self.env.observation_space
        return observation_space["observation"].shape[0]

    def get_goal_dim(self):
        observation_space = self.env.observation_space
        return observation_space["desired_goal"].shape[0]


class NormObs(ObservationWrapper, BasicWrapper):
    def __init__(self, env):
        super(NormObs, self).__init__(env)

    def observation(self, obs):
        return (obs - (self.maze_size / 2)) / self.maze_size
