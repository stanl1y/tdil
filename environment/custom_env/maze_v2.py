import gym
from gym import spaces
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import flowiz as fz


class Maze_v2(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, maze_size=100, random_reset=True, combine_s_g=False):
        super(Maze_v2, self).__init__()

        self.maze_size = maze_size
        self.random_reset = random_reset
        self.combine_s_g = combine_s_g  # combine state and goal or not
        self.max_episode_steps = 400
        # left, up, right, down
        self.ACTIONS = [
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, -1]),
        ]

        self.action_space = spaces.Discrete(4)
        if self.combine_s_g:
            self.observation_space = spaces.Box(
                low=-maze_size, high=maze_size, shape=(2,), dtype=np.int32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=maze_size, shape=(4,), dtype=np.int32
            )

    def is_terminal(self):
        return self.current_goal == 7

    def expert_step(self):
        target = self.target_list[self.current_goal]
        # if self.current_goal==1 and self.state[1]<51:#still in the original room
        #     target=np.array([24 if target[0]<=24 else 25, 51])
        # elif self.current_goal==2 and self.state[0]<51:
        #     target=np.array([51, 74 if target[1]<=74 else 75])
        # elif self.current_goal==3 and self.state[1]>48:
        #     target=np.array([74 if target[0]<=74 else 75, 48])

        dist = target - self.state
        if dist[0] > 0:
            ver_action = 2
        else:
            ver_action = 0

        if dist[1] > 0:
            hor_action = 1
        else:
            hor_action = 3

        if abs(dist[0]) > abs(dist[1]):
            if self.check_legal(self.state + self.ACTIONS[ver_action]):
                action = ver_action
            else:
                action = hor_action
        else:
            if self.check_legal(self.state + self.ACTIONS[hor_action]):
                action = hor_action
            else:
                action = ver_action

        return action

    def check_legal(self, state):
        x, y = state
        if (
            49 <= x and x <= 50 and not (74 <= y and y <= 75 and self.current_goal == 3)
        ):  # check horizontal wall
            return False
        if (
            49 <= y
            and y <= 50
            and not (24 <= x and x <= 25 and self.current_goal == 1)
            and not (74 <= x and x <= 75 and self.current_goal == 5)
        ):  # check vertical wall
            return False
        if x < 0 or x >= self.maze_size or y < 0 or y >= self.maze_size:
            return False

        return True

    def check_goal(self, state):
        return (state == self.target_list[self.current_goal]).all()

    def step(self, action):
        next_state = (np.array(self.state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if not self.check_legal(
            [x, y]
        ):  # x < 0 or x >= self.maze_size or y < 0 or y >= self.maze_size:
            next_state = self.state

        self.state = np.array(next_state)
        reward = 0.0
        if self.check_goal(self.state):
            self.current_goal += 1
            reward = 1.0

        # if self.is_small_reward(self.state):
        #     return self.state, 0.01, True, {}
        if self.combine_s_g:
            return (
                (self.state - self.target_list[self.current_goal]),
                reward,
                self.is_terminal(),
                {
                    "x": self.state[0],
                    "y": self.state[1],
                    "goal_idx": self.current_goal,
                    "goal": self.target_list[self.current_goal],
                },
            )
        else:
            return (
                np.concatenate((self.state, self.target_list[self.current_goal])),
                reward,
                self.is_terminal(),
                {
                    "x": self.state[0],
                    "y": self.state[1],
                    "goal_idx": self.current_goal,
                    "goal": self.target_list[self.current_goal],
                },
            )

    def reset(self):
        if self.random_reset:
            self.state = np.array(
                [np.random.randint(49), np.random.randint(49)]
            )  # (0~48,0~48)
            key1 = np.array([np.random.randint(2, 48), np.random.randint(2, 48)])
            while (key1 == self.state).all():
                key1 = np.array([np.random.randint(2, 48), np.random.randint(2, 48)])
            key2 = np.array([np.random.randint(2, 48), np.random.randint(52, 99)])
            bridge1 = np.array([24 if key2[0] <= 24 else 25, 51])
            key3 = np.array([np.random.randint(52, 99), np.random.randint(52, 99)])
            bridge2 = np.array([51, 74 if key3[1] <= 74 else 75])
            key4 = np.array([np.random.randint(52, 99), np.random.randint(2, 48)])
            bridge3 = np.array([74 if key4[0] <= 74 else 75, 48])
            self.target_list = [key1, bridge1, key2, bridge2, key3, bridge3, key4]

        else:
            self.state = np.array([48, 0])
            self.target_list = [
                np.array([20, 20]),  # key1
                np.array([25, 51]),  # bridge1
                np.array([30, 80]),  # key2
                np.array([51, 74]),  # bridge2
                np.array([70, 65]),  # key3
                np.array([75, 48]),  # bridge3
                np.array([80, 25]),  # key4
            ]
        self.current_goal = 0
        self.target_list.append(self.target_list[-1])
        if self.combine_s_g:
            return self.state - self.target_list[self.current_goal]
        else:
            return np.concatenate((self.state, self.target_list[self.current_goal]))

    def update_color(self, color, color_step):
        color += color_step
        mask = color > 255
        color[mask] = 255
        color_step[mask] *= -1

        mask = color < 0
        color[mask] = 0
        color_step[mask] *= -1

    def draw_border_v2(self, img, target_idx):
        mask = np.ones(img.shape)
        mask[49:51, :] = 0
        mask[:, 49:51] = 0
        mask[24:26, 49:51] = 1
        mask[49:51, 74:76] = 1
        mask[74:76, 49:51] = 1
        mask[20, 20] = 0
        mask[30, 80] = 0
        mask[70, 65] = 0
        mask[80, 25] = 0
        if target_idx == 0 or target_idx == 1:
            mask[51:, :] = 0
            mask[:, 51:] = 0
        elif target_idx == 2 or target_idx == 3:
            mask[51:, :] = 0
            mask[:, :49] = 0
        elif target_idx == 4 or target_idx == 5:
            mask[:49, :] = 0
            mask[:, :49] = 0
        elif target_idx == 6:
            mask[:49, :] = 0
            mask[:, 51:] = 0

        img = img * mask
        return img

    def eval_toy_q(self, agent, NeighborhoodNet, storage, path, episode, oracle_neighbor):
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
        target_list = [
            np.array([20, 20]),  # key1
            np.array([25, 51]),  # bridge1
            np.array([30, 80]),  # key2
            np.array([51, 74]),  # bridge2
            np.array([70, 65]),  # key3
            np.array([75, 48]),  # bridge3
            np.array([80, 25]),  # key4
        ]

        """
        test episode
        """
        maze = np.ones((self.maze_size, self.maze_size, 3))
        maze *= 255
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
        rewards  # /=episode_steps
        with open(f"{path}raw_path/{episode}.pkl", "wb") as fp:
            pickle.dump(raw_path, fp)
        with open(f"{path}testing_reward/{episode}.pkl", "wb") as fp:
            pickle.dump(rewards, fp)
        maze /= 255.0
        maze = self.draw_border_v2(maze, -1)
        plt.imsave(f"{path}path/{episode}.png", maze)
        for target_idx, target in enumerate(target_list):
            """
            draw q-function for different target
            """
            if self.combine_s_g:
                query_np = np.concatenate(
                    (yy[:, :, None] - target[0], xx[:, :, None] - target[1]), axis=2
                ).reshape(self.maze_size * self.maze_size, 2)
            else:
                query_np = np.concatenate(
                    (yy[:, :, None], xx[:, :, None]), axis=2
                ).reshape(self.maze_size * self.maze_size, 2, 2)
                query_np = np.c_[
                    query_np, np.repeat([target], query_np.shape[0], axis=0)
                ]
            query_norm_np = (query_np - self.maze_size / 2) / self.maze_size
            query_norm = torch.FloatTensor(query_norm_np).cuda()
            with torch.no_grad():
                if hasattr(agent, "q_network"):
                    q = agent.q_network(query_norm)
                else:
                    q0 = agent.critic[0](query_norm)
                    q1 = agent.critic[1](query_norm)
                    q = torch.min(q0, q1)
                u = (
                    (q[:, 1] - q[:, 3]).cpu().numpy()
                )  # horizontal positive:right,  negative:left
                v = (
                    (q[:, 2] - q[:, 0]).cpu().numpy()
                )  # vertical positive:down negative:up
            tmp = np.zeros((self.maze_size, self.maze_size, 2))
            tmp[:, :, 0] = u.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]
            tmp[:, :, 1] = v.reshape(self.maze_size, self.maze_size, 1)[:, :, 0]

            with open(f"{path}raw_flow/{episode}.flo", "wb") as fp:
                pickle.dump(tmp, fp)
            img = fz.convert_from_flow(tmp)
            img = self.draw_border_v2(img, target_idx)
            img = img / 255
            plt.imsave(f"{path}color/{episode}-{target_idx}.png", img)
            # add path to q image
            # mask=maze.mean(axis=2)>0
            # img[mask]=maze[mask]
            # maze/=255.0
            # # maze=draw_border(maze,-1)
            # plt.imsave(f"{path}path/{episode}.png",maze)

            plt.gca().invert_yaxis()
            plt.imshow(img)
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
            plt.savefig(f"{path}arr/{episode}-{target_idx}.png", pad_inches=0)
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
            neighborhood_reward = self.draw_border_v2(neighborhood_reward, target_idx)
            neighborhood_reward = neighborhood_reward / neighborhood_reward.max()
            neighborhood_reward = np.repeat(neighborhood_reward, 3, axis=2)
            plt.imsave(
                f"{path}neighbor_reward/{episode}-{target_idx}.png", neighborhood_reward
            )

        return img, arr_img, maze, rewards
