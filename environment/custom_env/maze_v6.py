import gym
from gym import spaces
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import flowiz as fz
import copy
import torch.nn.functional as F


class Maze_v6(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, maze_size=10, random_reset=False, reward_type="gail"):
        super(Maze_v6, self).__init__()

        self.maze_size = maze_size
        self.state = np.asarray((self.maze_size - 1, 0))
        self.random_reset = random_reset
        self.agent_step = 0
        self.max_episode_steps = 50

        # left, up, right, down
        self.ACTIONS = [
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, -1]),
        ]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=maze_size, shape=(2,), dtype=np.uint8
        )
        self.maze = np.ones((self.maze_size, self.maze_size))
        self.maze[2, 1:5] = 2
        self.maze[2:6, 4] = 2
        self.maze[5, 4:7] = 2
        self.maze[5:, 6] = 2
        self.maze[9, 6:] = 2
        self.expert_states = np.argwhere(self.maze == 2)
        # sort according second item of each row
        self.expert_states = self.expert_states[self.expert_states[:, 1].argsort()]

        self.expert_states_set = set(tuple(x) for x in self.expert_states)
        self.expert_transitions = []

        expert_init_state = (2, 1)
        self.expert_states_set.remove(expert_init_state)
        while not len(self.expert_states_set) == 0:
            for s in self.expert_states:
                s = (s[0], s[1])
                if s in self.expert_states_set:
                    for idx, a in enumerate(self.ACTIONS):
                        new_state = tuple(expert_init_state + a)
                        if s == new_state:
                            self.expert_transitions.append((expert_init_state, idx, s))
                            expert_init_state = s
                            break
                    if expert_init_state == s:
                        break
            self.expert_states_set.remove(expert_init_state)
            # for item in self.expert_transitions:
            #     print(item[0])
            # print("-------------------")

        self.expert_state_action_set = set()  # for gail reward
        self.expert_state_action_list = []
        for item in self.expert_transitions:
            self.expert_state_action_set.add(
                (
                    item[0][0],
                    item[0][1],
                    self.ACTIONS[item[1]][0],
                    self.ACTIONS[item[1]][1],
                )
            )
            self.expert_state_action_list.append(
                np.array(
                    [
                        item[0][0],
                        item[0][1],
                        self.ACTIONS[item[1]][0],
                        self.ACTIONS[item[1]][1],
                    ]
                )
            )
        self.expert_state_action_list = np.array(self.expert_state_action_list)
        if not os.path.exists(
            "./saved_expert_transition/Maze-v6/oracle/episode_num1.pkl"
        ):
            with open(
                "./saved_expert_transition/Maze-v6/oracle/episode_num1.pkl", "wb"
            ) as f:
                data = {}
                data["states"] = np.array([item[0] for item in self.expert_transitions])
                data["states"] = (
                    data["states"] - (self.maze_size / 2)
                ) / self.maze_size
                data["actions"] = np.array(
                    [[item[1]] for item in self.expert_transitions]
                )
                data["next_states"] = np.array(
                    [item[2] for item in self.expert_transitions]
                )
                data["next_states"] = (
                    data["next_states"] - (self.maze_size / 2)
                ) / self.maze_size
                data["rewards"] = np.ones((len(self.expert_transitions), 1))
                data["dones"] = np.zeros((len(self.expert_transitions), 1))
                pickle.dump(data, f)
        # draw maze and expert trajectory
        tmp = np.zeros((self.maze_size, self.maze_size, 3))
        tmp[self.maze == 2] = [1.0, 1.0, 1.0]
        tmp = self.resize_and_draw_wall(tmp)
        plt.imsave(f"environment/custom_env/Maze-v6.png", tmp)

        if reward_type == "gail":
            self.reward_function = self.reward_function_gail
        elif reward_type == "l2":
            self.reward_function = self.reward_function_l2
        elif reward_type == "l2_s":
            self.reward_function = self.reward_function_l2_state
            xx, yy = np.meshgrid(range(self.maze_size), range(self.maze_size))
            query_np = np.concatenate((yy[:, :, None], xx[:, :, None]), axis=2).reshape(
                self.maze_size * self.maze_size, 2
            )
            reward_list = []
            for item in query_np:
                reward_list.append(self.reward_function(next_state=item))
            reward_list = np.array(reward_list).reshape(
                (self.maze_size, self.maze_size, 1)
            )
            reward_list = np.repeat(reward_list, 3, axis=2)
            reward_list = self.resize_and_draw_wall(reward_list)
            plt.imsave(f"environment/custom_env/Maze-v6_L2_s.png", reward_list)

    def is_terminal(self, state):
        x, y = state
        return (
            x == self.maze_size - 1 and y == self.maze_size - 1
        ) or self.max_episode_steps == self.agent_step

    def update_color(self, color, color_step):
        color += color_step
        mask = color > 255
        color[mask] = 255
        color_step[mask] *= -1

        mask = color < 0
        color[mask] = 0
        color_step[mask] *= -1

    def resize_and_draw_wall(self, maze):
        maze = np.repeat(maze, 4, axis=0)
        maze = np.repeat(maze, 4, axis=1)
        maze[3 * 4 :, 4 * 4] = [1, 0, 0]
        maze[4 * 4 : 9 * 4, 7 * 4] = [1, 0, 0]
        maze[9 * 4 - 1, 7 * 4 :] = [1, 0, 0]
        return maze

    def eval_toy_q(
        self,
        agent,
        path,
        step_num,
        storage=None,
        NeighborhoodNet=None,
        oracle_neighbor=False,
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
        if not os.path.exists(path + "raw_flow/"):
            os.makedirs(path + "raw_flow/")
        if not os.path.exists(path + "raw_path/"):
            os.makedirs(path + "raw_path/")
        if not os.path.exists(path + "testing_reward/"):
            os.makedirs(path + "testing_reward/")
        if NeighborhoodNet is not None:
            if not os.path.exists(path + "neighbor_reward/"):
                os.makedirs(path + "neighbor_reward/")
        xx, yy = np.meshgrid(range(self.maze_size), range(self.maze_size))
        query_np = np.concatenate((yy[:, :, None], xx[:, :, None]), axis=2).reshape(
            self.maze_size * self.maze_size, 2
        )
        query_norm_np = (query_np - self.maze_size / 2) / self.maze_size
        query_norm = torch.FloatTensor(query_norm_np).cuda()
        # with torch.no_grad():
        #     if hasattr(agent, "q_network"):
        #         q = agent.q_network(query_norm)
        #     else:
        #         q0 = agent.critic[0](query_norm)
        #         q1 = agent.critic[1](query_norm)
        #         q = torch.min(q0, q1)
        #     u = (
        #         (q[:, 1] - q[:, 3]).cpu().numpy()
        #     )  # horizontal positive:right,  negative:left
        #     v = (q[:, 2] - q[:, 0]).cpu().numpy()  # vertical positive:down negative:up
        with torch.no_grad():
            q = F.softmax(agent.actor.forward(query_norm), dim=1)
            u = (
                (q[:, 1] - q[:, 3]).cpu().numpy()
            )  # horizontal positive:right,  negative:left
            v = (q[:, 2] - q[:, 0]).cpu().numpy()  # vertical positive:down negative:up
        maze = np.zeros((self.maze_size, self.maze_size, 2))
        maze[:, :, 0] = u.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]
        maze[:, :, 1] = v.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]
        with open(f"{path}raw_flow/{step_num}.flo", "wb") as fp:
            pickle.dump(maze, fp)
        img = fz.convert_from_flow(maze)
        plt.imsave(f"{path}color/{step_num}.png", img)

        maze = np.zeros((self.maze_size, self.maze_size, 3))
        # color=np.random.randint(256,size=(3))
        # color_step=np.random.randint(-10,10,size=(3))
        color = np.array([128, 128, 128])
        # color_step = np.array([3, 2, -2])
        color_step = np.array([0, 0, 0])
        old_state = copy.deepcopy(self.state)
        old_agent_step = self.agent_step

        self.reset()
        self.state = np.array([self.maze_size - 1, 0])
        state = copy.deepcopy(self.state)
        state = (state - (self.maze_size / 2)) / self.maze_size

        maze[self.maze_size - 1, 0] = color
        self.update_color(color, color_step)
        done = False
        episode_steps = 0
        rewards = 0
        raw_path = []
        while not done:
            action = agent.act(state, testing=True)
            next_state, reward, done, info = self.step(action)
            rewards += reward
            pos = [info["x"], info["y"]]
            raw_path.append((pos[0], pos[1]))
            if not done:
                maze[pos[0], pos[1]] = color
                self.update_color(color, color_step)
            else:
                maze[pos[0], pos[1]] = np.array([1, 1, 1])
            state = next_state
            state = (state - (self.maze_size / 2)) / self.maze_size
            episode_steps += 1
        rewards /= episode_steps
        with open(f"{path}raw_path/{step_num}.pkl", "wb") as fp:
            pickle.dump(raw_path, fp)
        with open(f"{path}testing_reward/{step_num}.pkl", "wb") as fp:
            pickle.dump(rewards, fp)
        # add path to q image
        mask = maze.mean(axis=2) > 0
        img[mask] = maze[mask]
        maze /= 255.0
        maze = self.resize_and_draw_wall(maze)
        plt.imsave(f"{path}path/{step_num}.png", maze)

        plt.gca().invert_yaxis()
        # turn img to float array and divide by 255
        img = img.astype("float32") / 255.0

        img = self.resize_and_draw_wall(img)
        plt.imshow(img)
        u = u.reshape((self.maze_size, self.maze_size))
        v = v.reshape((self.maze_size, self.maze_size))
        xx, yy = np.meshgrid(
            np.arange(self.maze_size * 4, step=4), np.arange(self.maze_size * 4, step=4)
        )

        arr_img = plt.quiver(
            xx.reshape(-1) + 2,
            yy.reshape(-1) + 2,
            u.reshape(-1),
            v.reshape(-1) * -1,
        )
        plt.savefig(f"{path}arr/{step_num}.png", pad_inches=0)
        plt.close()
        # plt.imsave(f"{path}arr{episode}.png",arr_img)
        if NeighborhoodNet is not None:
            with torch.no_grad():
                neighborhood_reward = agent.neighborhood_reward(
                    NeighborhoodNet, storage, query_norm_np, oracle_neighbor
                )
            neighborhood_reward = neighborhood_reward.cpu().numpy()
            neighborhood_reward = neighborhood_reward.reshape(
                self.maze_size, self.maze_size, 1
            )
            new_neighborhood_reward = np.zeros((self.maze_size, self.maze_size,1))
            for i in range(self.maze_size):
                for j in range(self.maze_size):
                    if(self.check_walkable([i,j],[max(i - 1, 0), j])):
                        new_neighborhood_reward[i][j] = max(new_neighborhood_reward[i][j],neighborhood_reward[max(i - 1, 0), j])
                    if(self.check_walkable([i,j],[min(i + 1, self.maze_size - 1), j])):
                        new_neighborhood_reward[i][j] = max(new_neighborhood_reward[i][j],neighborhood_reward[min(i + 1, self.maze_size - 1), j])
                    if(self.check_walkable([i,j],[i, max(j - 1, 0)])):
                        new_neighborhood_reward[i][j] = max(new_neighborhood_reward[i][j],neighborhood_reward[i, max(j - 1, 0)])
                    if(self.check_walkable([i,j],[i, min(j + 1, self.maze_size - 1)])):
                        new_neighborhood_reward[i][j] = max(new_neighborhood_reward[i][j],neighborhood_reward[i, min(j + 1, self.maze_size - 1)])
            for s, _, _ in self.expert_transitions:
                new_neighborhood_reward[s[0], s[1]] = 3
            new_neighborhood_reward = new_neighborhood_reward / new_neighborhood_reward.max()
            new_neighborhood_reward = np.repeat(new_neighborhood_reward, 3, axis=2)
            new_neighborhood_reward = self.resize_and_draw_wall(new_neighborhood_reward)

            plt.imsave(f"{path}neighbor_reward/{step_num}.png", new_neighborhood_reward)
        self.state = old_state
        self.agent_step = old_agent_step
        return img, arr_img, maze, episode_steps

    def reward_function_gail(self, state=None, action=None, next_state=None, **kwargs):
        if (state[0], state[1], action[0], action[1]) in self.expert_state_action_set:
            return 1
        else:
            return 0

    def reward_function_l2(self, state=None, action=None, next_state=None, **kwargs):
        tmp = np.array([state[0], state[1], action[0], action[1]])
        # calc l2 distance with each row in expert_state_action_list
        l2 = np.linalg.norm(self.expert_state_action_list - tmp, axis=1)
        # get the min l2 distance
        min_l2 = np.min(l2)
        # turn into reward
        reward = np.exp(-min_l2 / 10)
        return reward

    def reward_function_l2_state(
        self, state=None, action=None, next_state=None, **kwargs
    ):
        # calc l2 distance with each row in expert_state_action_list
        tmp = self.expert_state_action_list[:, :2]
        l2 = np.linalg.norm(tmp - next_state, axis=1)
        # get the min l2 distance
        min_l2 = np.min(l2)
        # turn into reward
        reward = np.exp(-min_l2)
        return reward

    def check_walkable(self, s1, s2):
        hor_set = {s1[1], s2[1]}
        ver_set = {s1[0], s2[0]}
        if (
            (s1[0] >= 3 and (3 in hor_set) and (4 in hor_set))
            or (s1[0] >= 4 and s1[0] < 9 and (6 in hor_set) and (7 in hor_set))
            or (s1[1] >= 7 and (8 in ver_set) and (9 in ver_set))
        ):
            return False
        return True

    def step(self, action):
        self.agent_step += 1
        next_state = (np.array(self.state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if (
            x < 0
            or x >= self.maze_size
            or y < 0
            or y >= self.maze_size
            or not self.check_walkable(self.state, next_state)
        ):
            next_state = self.state

        reward = self.reward_function(
            state=self.state, action=self.ACTIONS[action], next_state=next_state
        )
        self.state = np.array(next_state)

        return (
            self.state,
            reward,
            self.is_terminal(self.state),
            {
                "x": self.state[0],
                "y": self.state[1],
                "agent_step_total_penalty": self.agent_step,
            },
        )

    def reset(self):
        # if self.random_reset:
        #     self.state = np.asarray(
        #         (
        #             np.random.randint(self.maze_size // 2, self.maze_size),
        #             np.random.randint(self.maze_size // 2),
        #         )
        #     )
        # else:
        #     self.state = np.asarray((self.maze_size - 1, 0))
        self.state = np.asarray(
            (
                np.random.randint(0, self.maze_size),
                np.random.randint(0, self.maze_size),
            )
        )
        self.agent_step = 0
        return self.state
