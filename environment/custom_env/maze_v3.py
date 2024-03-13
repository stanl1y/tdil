import gym
from gym import spaces
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import flowiz as fz


class Maze_v3(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, maze_size=50, random_reset=False):
        super(Maze_v3, self).__init__()

        self.maze_size = maze_size
        self.state = np.asarray((self.maze_size - 1, 0))
        self.random_reset = random_reset
        self.agent_step = 0
        self.max_episode_steps = 250

        # up, right, down, right-up, right-down
        self.ACTIONS = [
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([-1, 1]),
            np.array([1, 1]),
        ]

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=maze_size, shape=(2,), dtype=np.uint8
        )
        self.maze = np.ones((self.maze_size, self.maze_size))
        # define the location of walls [-20:,24], [-30:-9,30]
        self.maze[-19:, 24] = 0
        self.maze[-30:-9, 30] = 0  # -30 to -10
        # define expert's path
        # self.maze[-5:, 20] = 2
        self.maze[-10:-5, 21] = 2  # -10 to -6
        self.maze[-15:-10, 22] = 2  # -15 to -11
        self.maze[-19:-15, 23] = 2  # -20 to -16
        self.maze[-20, 24] = 2
        self.maze[-19:-17, 25] = 2
        self.maze[-17:-15, 26] = 2
        self.maze[-15:-13, 27] = 2
        self.maze[-13:-11, 28] = 2
        self.maze[-11:-9, 29] = 2
        self.maze[-9, 30] = 2
        self.maze[-8, 31] = 2
        self.maze[-7, 32] = 2
        self.maze[-6, 33] = 2
        self.maze[-5, 34] = 2
        self.maze[-4, 35] = 2
        self.maze[-3, 36] = 2
        self.maze[-2, 37] = 2
        self.maze[-1, 38:] = 2
        self.expert_states = np.argwhere(self.maze == 2)
        self.expert_states_set = set(tuple(x) for x in self.expert_states)
        # self.expert_states_set.remove((49,49))
        self.expert_transitions = []
        for state in self.expert_states_set:
            if state == (49, 49):
                continue
            for idx, action in enumerate(self.ACTIONS):
                if state[1] < 24 and action[0] == 1:
                    continue
                if state[1] >= 24 and action[0] == -1:
                    continue
                # next_state = tuple(np.clip(state + action, 0, self.maze_size - 1))
                next_state = tuple(state + action)
                if next_state in self.expert_states_set:
                    self.expert_transitions.append((state, idx, next_state))
        self.expert_transitions.sort(key=lambda x: x[0][1])
        self.expert_state_action_set = set()
        for item in self.expert_transitions:
            self.expert_state_action_set.add(
                (item[0][0], item[0][1], self.ACTIONS[item[1]][0], self.ACTIONS[item[1]][1])
            )
        # dump expert data
        # if expert data don't exists
        if not os.path.exists(
            "./saved_expert_transition/Maze-v3/oracle/episode_num1.pkl"
        ):
            with open(
                "./saved_expert_transition/Maze-v3/oracle/episode_num1.pkl", "wb"
            ) as f:
                data = {}
                data["states"] = np.array([item[0] for item in self.expert_transitions])
                data["actions"] = np.array(
                    [item[1] for item in self.expert_transitions]
                )
                data["next_states"] = np.array(
                    [item[2] for item in self.expert_transitions]
                )
                data["rewards"] = np.ones(len(self.expert_transitions), 1)
                data["dones"] = np.zeros(len(self.expert_transitions), 1)
                pickle.dump(data, f)

    def is_terminal(self, state):
        x, y = state
        return (
            x == self.maze_size - 1
            and y == self.maze_size - 1
            or self.max_episode_steps == self.agent_step
        )

    def is_small_reward(self, state):
        x, y = state
        return x >= 3 and x <= self.maze_size - 4 and y >= 3 and y <= self.maze_size - 4

    def update_color(self, color, color_step):
        color += color_step
        mask = color > 255
        color[mask] = 255
        color_step[mask] *= -1

        mask = color < 0
        color[mask] = 0
        color_step[mask] *= -1

    def eval_toy_q(
        self, agent, NeighborhoodNet, storage, path, episode, oracle_neighbor=False
    ):
        """
        0:up, 1:right, 2:down, 3:left
        """
        if not os.path.exists(path + "color/"):
            os.makedirs(path + "color/")
        if not os.path.exists(path + "arr/"):
            os.makedirs(path + "arr/")
        if not os.path.exists(path + "path/"):
            os.makedirs(path + "path/")
        if not os.path.exists(path + "neighbor_reward/"):
            os.makedirs(path + "neighbor_reward/")
        if not os.path.exists(path + "raw_flow/"):
            os.makedirs(path + "raw_flow/")
        if not os.path.exists(path + "raw_path/"):
            os.makedirs(path + "raw_path/")
        if not os.path.exists(path + "testing_reward/"):
            os.makedirs(path + "testing_reward/")
        xx, yy = np.meshgrid(range(self.maze_size), range(self.maze_size))
        query_np = np.concatenate((yy[:, :, None], xx[:, :, None]), axis=2).reshape(
            self.maze_size * self.maze_size, 2
        )
        query_norm_np = (query_np - self.maze_size / 2) / self.maze_size
        query_norm = torch.FloatTensor(query_norm_np).cuda()
        with torch.no_grad():
            if hasattr(agent, "q_network"):
                q = agent.q_network(query_norm)
            else:
                q0 = agent.critic[0](query_norm)
                q1 = agent.critic[1](query_norm)
                q = torch.min(q0, q1)
            # u = (
            #     (q[:, 1] - q[:, 3]).cpu().numpy()
            # )  # horizontal positive:right,  negative:left
            # v = (q[:, 2] - q[:, 0]).cpu().numpy()  # vertical positive:down negative:up
            highest_action = torch.argmax(q, dim=1).cpu().numpy()
            # action 0:v=-1,u=0; 1:v=0,u=1; 2:v=1,u=0; 3:v=-0.5,u=0.5; 4:v=0.5,u=0.5
            v = np.zeros_like(highest_action)
            u = np.zeros_like(highest_action)
            v[highest_action == 0] = -1
            v[highest_action == 1] = 0
            v[highest_action == 2] = 1
            v[highest_action == 3] = -0.5
            v[highest_action == 4] = 0.5
            u[highest_action == 0] = 0
            u[highest_action == 1] = 1
            u[highest_action == 2] = 0
            u[highest_action == 3] = 0.5
            u[highest_action == 4] = 0.5

        # maze = np.zeros((self.maze_size, self.maze_size, 2))
        # maze[:, :, 0] = u.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]
        # maze[:, :, 1] = v.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]

        # with open(f"{path}raw_flow/{episode}.flo", "wb") as fp:
        #     pickle.dump(maze, fp)
        # img = fz.convert_from_flow(maze)
        # plt.imsave(f"{path}color/{episode}.png", img)

        maze = np.zeros((self.maze_size, self.maze_size, 3))
        # color=np.random.randint(256,size=(3))
        # color_step=np.random.randint(-10,10,size=(3))
        color = np.array([128, 128, 128])
        color_step = np.array([3, 2, -2])
        state = self.reset()
        state = (state - (self.maze_size / 2)) / self.maze_size

        maze[self.maze_size - 1, 0] = color
        self.update_color(color, color_step)
        done = False
        episode_steps = 0
        rewards = 0
        raw_path = []
        while not done and episode_steps < self.max_episode_steps:
            action = agent.act(state, testing=True)
            next_state, reward, done, info = self.step(action)
            rewards += reward
            pos = [info["x"], info["y"]]
            raw_path.append((pos[0], pos[1]))
            if not done:
                maze[pos[0], pos[1]] = color
                self.update_color(color, color_step)
            else:
                maze[pos[0], pos[1]] = np.array([255, 0, 0])
            state = next_state
            state = (state - (self.maze_size / 2)) / self.maze_size
            episode_steps += 1
        rewards /= episode_steps
        with open(f"{path}raw_path/{episode}.pkl", "wb") as fp:
            pickle.dump(raw_path, fp)
        with open(f"{path}testing_reward/{episode}.pkl", "wb") as fp:
            pickle.dump(rewards, fp)
        # add path to q image
        # mask = maze.mean(axis=2) > 0
        # img[mask] = maze[mask]
        maze /= 255.0
        plt.imsave(f"{path}path/{episode}.png", maze)

        # plt.gca().invert_yaxis()
        # plt.imshow(img)
        u = u.reshape((self.maze_size, self.maze_size))
        v = v.reshape((self.maze_size, self.maze_size))
        subxx, subyy = np.meshgrid(
            range(self.maze_size // 4), range(self.maze_size // 4)
        )
        subxx *= 4
        subyy *= 4
        subu = np.zeros(subxx.shape)
        subv = np.zeros(subxx.shape)
        subu = (
            subu
            + u[subyy, subxx]
            + u[subyy + 1, subxx]
            + u[subyy + 2, subxx]
            + u[subyy + 3, subxx]
            + u[subyy, subxx + 1]
            + u[subyy + 1, subxx + 1]
            + u[subyy + 2, subxx + 1]
            + u[subyy + 3, subxx + 1]
            + u[subyy, subxx + 2]
            + u[subyy + 1, subxx + 2]
            + u[subyy + 2, subxx + 2]
            + u[subyy + 3, subxx + 2]
            + u[subyy, subxx + 3]
            + u[subyy + 1, subxx + 3]
            + u[subyy + 2, subxx + 3]
            + u[subyy + 3, subxx + 3]
        )
        subv = (
            subv
            + v[subyy, subxx]
            + v[subyy + 1, subxx]
            + v[subyy + 2, subxx]
            + v[subyy + 3, subxx]
            + v[subyy, subxx + 1]
            + v[subyy + 1, subxx + 1]
            + v[subyy + 2, subxx + 1]
            + v[subyy + 3, subxx + 1]
            + v[subyy, subxx + 2]
            + v[subyy + 1, subxx + 2]
            + v[subyy + 2, subxx + 2]
            + v[subyy + 3, subxx + 2]
            + v[subyy, subxx + 3]
            + v[subyy + 1, subxx + 3]
            + v[subyy + 2, subxx + 3]
            + v[subyy + 3, subxx + 3]
        )  # arr_img=plt.quiver(yy.reshape(-1),xx.reshape(-1),u,v)
        arr_img = plt.quiver(
            subxx.reshape(-1),
            subyy.reshape(-1),
            subu.reshape(-1),
            subv.reshape(-1) * -1,
        )
        plt.savefig(f"{path}arr/{episode}.png", pad_inches=0)
        plt.close()
        # plt.imsave(f"{path}arr{episode}.png",arr_img)
        with torch.no_grad():
            neighborhood_reward = agent.neighborhood_reward(
                NeighborhoodNet, storage, query_norm_np, oracle_neighbor
            )
        neighborhood_reward = neighborhood_reward.cpu().numpy()
        neighborhood_reward = neighborhood_reward.reshape(
            self.maze_size, self.maze_size, 1
        )
        neighborhood_reward = neighborhood_reward / neighborhood_reward.max()
        neighborhood_reward = np.repeat(neighborhood_reward, 3, axis=2)
        plt.imsave(f"{path}neighbor_reward/{episode}.png", neighborhood_reward)
        return arr_img, maze, rewards

    # def expert_step(self):
    #     x, y = self.state
    #     if x > self.maze_size - 1 - y:
    #         action = 0
    #     else:
    #         action = 1
    #     return action

    def reward_function(self, state, action):
        if (state[0], state[1], action[0], action[1]) in self.expert_state_action_set:
            return 1
        else:
            return 0

    def step(self, action):
        self.agent_step += 1
        next_state = (np.array(self.state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if (
            x < 0
            or x >= self.maze_size
            or y < 0
            or y >= self.maze_size
            or self.maze[x, y] == 0
        ):
            next_state = self.state

        self.state = np.array(next_state)

        # if self.is_small_reward(self.state):
        #     return self.state, 0.01, True, {}

        reward = self.reward_function(self.state, self.ACTIONS[action])
        if self.is_terminal(self.state):
            return (
                self.state,
                reward,
                True,
                {
                    "x": self.state[0],
                    "y": self.state[1],
                    "agent_step_penalty": self.agent_step,
                },
            )
        return self.state, reward, False, {"x": self.state[0], "y": self.state[1]}

    def reset(self):
        if self.random_reset:
            self.state = np.asarray(
                (
                    np.random.randint(self.maze_size // 2, self.maze_size),
                    np.random.randint(self.maze_size // 2),
                )
            )
        else:
            self.state = np.asarray((self.maze_size - 1, 0))
        self.agent_step = 0
        return self.state
