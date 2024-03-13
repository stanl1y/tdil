import numpy as np
import os
import pickle


class her_replay_buffer:
    """
    her replay buffer stores experiences trajectory wise
    assuming each trajectory has 1000 steps maximum
    """

    def __init__(self, size, state_dim, goal_dim, action_dim, reward_fn):
        self.rollout_num = size // 1000
        self.storage_index = 0
        self.states = np.empty((self.rollout_num, 1000, state_dim))
        self.achieved_goal = np.empty((self.rollout_num, 1000, goal_dim))
        self.actions = np.empty((self.rollout_num, 1000, action_dim))
        self.rewards = np.empty((self.rollout_num, 1000, 1))
        self.next_states = np.empty((self.rollout_num, 1000, state_dim))
        self.next_achieved_goal = np.empty((self.rollout_num, 1000, goal_dim))
        self.desired_goal = np.empty((self.rollout_num, 1000, goal_dim))
        self.dones = np.empty((self.rollout_num, 1000, 1))
        self.rollout_size = np.empty((self.rollout_num))
        self.step_idx = 0
        self.reward_function = reward_fn

    def store(self, s, a, r, ss, d):
        index = self.storage_index % self.rollout_num
        self.states[index][self.step_idx] = s["observation"]
        self.achieved_goal[index][self.step_idx] = s["achieved_goal"]
        self.actions[index][self.step_idx] = a
        self.rewards[index][self.step_idx] = r
        self.next_states[index][self.step_idx] = ss["observation"]
        self.next_achieved_goal[index][self.step_idx] = ss["achieved_goal"]
        self.desired_goal[index][self.step_idx] = s["desired_goal"]
        self.dones[index][self.step_idx] = d
        if d or self.step_idx == self.states.shape[1] - 1:
            self.rollout_size[index] = self.step_idx + 1
            self.step_idx = 0
            self.storage_index += 1
        else:
            self.step_idx += 1

    def clear(self):
        self.storage_index = 0

    def sample(self, batch_size, future_ratio=4):
        """
        code from https://github.com/TianhongDai/hindsight-experience-replay/blob/master/her_modules/her.py
        """
        rollout_idxs = np.random.randint(
            0, min(self.storage_index, self.rollout_num), batch_size
        )
        T = self.rollout_size[rollout_idxs]
        t_samples = np.random.randint(T)
        future_p = 1 - (1.0 / (1 + future_ratio))
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        states = self.states[rollout_idxs, t_samples]
        actions = self.actions[rollout_idxs, t_samples]
        next_states = self.next_states[rollout_idxs, t_samples]
        next_achieved_goal = self.next_achieved_goal[rollout_idxs, t_samples]
        desired_goal = self.desired_goal[rollout_idxs, t_samples]
        dones = self.dones[rollout_idxs, t_samples] * 0 #all HER done is false

        future_achieved_goal = self.achieved_goal[rollout_idxs[her_indexes], future_t]
        desired_goal[her_indexes] = future_achieved_goal
        rewards = np.expand_dims(
            self.reward_function(next_achieved_goal, desired_goal, None), 1
        )
        states = np.append(states, desired_goal,axis=1)
        next_states = np.append(next_states, desired_goal,axis=1)

        return (states, actions, rewards, next_states, dones)

    def write_storage(
        self, based_on_transition_num, expert_data_num, algo, env_id, data_name=""
    ):
        if data_name != "":
            data_name = f"_{data_name}"
        path = f"./saved_expert_transition/{env_id}/{algo}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        save_idx = min(self.storage_index, self.rollout_num)
        print(save_idx)
        data = {
            "states": self.states[:save_idx],
            "actions": self.actions[:save_idx],
            "rewards": self.rewards[:save_idx],
            "next_states": self.next_states[:save_idx],
            "dones": self.dones[:save_idx],
        }
        if based_on_transition_num:
            file_name = f"transition_num{expert_data_num}{data_name}.pkl"
        else:
            file_name = f"episode_num{expert_data_num}{data_name}.pkl"
        print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), "wb") as handle:
            pickle.dump(data, handle)

    def load_expert_data(self, algo, env_id, duplicate_expert_last_state, data_name=""):
        if data_name == "":
            path = f"./saved_expert_transition/{env_id}/{algo}/"
            if not os.path.isdir(path):
                path = f"./saved_expert_transition/{env_id}/oracle/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[0]
        else:
            path = f"./saved_expert_transition/{env_id}/{data_name}.pkl"
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        print(f"load expert data from path : {path}")
        self.expert_states = data["states"]
        self.expert_actions = data["actions"]
        self.expert_rewards = data["rewards"]
        self.expert_next_states = data["next_states"]
        self.expert_dones = data["dones"]
        if duplicate_expert_last_state:
            done_idx = np.argwhere(self.expert_dones.reshape(-1) == 1).reshape(-1)
            done_expert_states = self.expert_states[done_idx]
            done_expert_actions = self.expert_actions[done_idx]
            done_expert_rewards = self.expert_rewards[done_idx]
            done_expert_next_states = self.expert_next_states[done_idx]
            done_expert_dones = self.expert_dones[done_idx]
            for _ in range(5):
                self.expert_states = np.concatenate(
                    (
                        self.expert_states,
                        done_expert_states,
                    ),
                    axis=0,
                )
                self.expert_actions = np.concatenate(
                    (
                        self.expert_actions,
                        done_expert_actions,
                    ),
                    axis=0,
                )
                self.expert_rewards = np.concatenate(
                    (
                        self.expert_rewards,
                        done_expert_rewards,
                    ),
                    axis=0,
                )
                self.expert_next_states = np.concatenate(
                    (
                        self.expert_next_states,
                        done_expert_next_states,
                    ),
                    axis=0,
                )
                self.expert_dones = np.concatenate(
                    (
                        self.expert_dones,
                        done_expert_dones,
                    ),
                    axis=0,
                )

    def __len__(self):
        return min(self.storage_index, self.rollout_num)
