import numpy as np
import os
import pickle
import torch


class normal_replay_buffer:
    def __init__(
        self,
        size,
        state_dim,
        action_dim,
        save_env_states=False,
        save_state_idx=False,
        to_tensor=False,
    ):
        self.size = size
        self.storage_index = 0
        self.states = np.empty((size, state_dim))
        self.actions = np.empty((size, action_dim))
        self.rewards = np.empty((size, 1))
        self.next_states = np.empty((size, state_dim))
        self.dones = np.empty((size, 1))
        self.env_states = [] if save_env_states else None
        self.states_idx = np.empty((size, 1), dtype=int) if save_state_idx else None
        self.to_tensor = to_tensor
        if self.to_tensor:
            print("Replay buffer stores tensor")
            self.states = torch.empty((size, state_dim))
            self.actions = torch.empty((size, action_dim))
            self.rewards = torch.empty((size, 1))
            self.next_states = torch.empty((size, state_dim))
            self.dones = torch.empty((size, 1))

    def store(self, s, a, r, ss, d, env_state=None, state_idx=None):
        if self.to_tensor:
            s = torch.FloatTensor(s)
            a = torch.FloatTensor(a)
            r = torch.FloatTensor([r])
            ss = torch.FloatTensor(ss)
            d = torch.FloatTensor([d])
        index = self.storage_index % self.size
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.next_states[index] = ss
        self.dones[index] = d
        if env_state is not None:
            self.env_states.append(env_state)
        if state_idx is not None:
            self.states_idx[index] = state_idx
        self.storage_index += 1

    def clear(self):
        self.storage_index = 0

    def sample(
        self,
        batch_size,
        expert=False,
        return_idx=False,
        return_expert_env_states=False,
        exclude_tail_num=0,
        only_last_1m=True,
    ):
        if expert:
            if batch_size == -1:
                indices = np.random.permutation(
                    len(self.expert_states) - exclude_tail_num
                )
            else:
                indices = np.random.choice(
                    len(self.expert_states) - exclude_tail_num, batch_size
                )
            tmp = (
                self.expert_states[indices],
                self.expert_actions[indices],
                self.expert_rewards[indices],
                self.expert_next_states[indices],
                self.expert_dones[indices],
            )
            if return_expert_env_states:
                tmp = tmp + (self.expert_env_states[indices.item()],)
            if return_idx:
                tmp = tmp + (indices,)
        else:
            if only_last_1m and self.size > 1000000:
                indices = np.random.randint(
                    max(0, self.storage_index - 1000000),
                    self.storage_index,
                    size=batch_size,
                )
            else:
                indices = np.random.randint(
                    min(self.storage_index, self.size), size=batch_size
                )
            tmp = (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
            )
            if return_idx:
                tmp = tmp + (self.states_idx[indices],)
        return tmp

    def write_storage(
        self, based_on_transition_num, expert_data_num, algo, env_id, data_name=""
    ):
        if self.to_tensor:
            self.states = self.states.numpy()
            self.actions = self.actions.numpy()
            self.rewards = self.rewards.numpy()
            self.next_states = self.next_states.numpy()
            self.dones = self.dones.numpy()
        if data_name != "":
            data_name = f"_{data_name}"
        path = f"./saved_expert_transition/{env_id}/{algo}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        save_idx = min(self.storage_index, self.size)
        print(save_idx)
        data = {
            "states": self.states[:save_idx],
            "actions": self.actions[:save_idx],
            "rewards": self.rewards[:save_idx],
            "next_states": self.next_states[:save_idx],
            "dones": self.dones[:save_idx],
            "env_states": self.env_states[:save_idx] if self.env_states else None,
        }
        if based_on_transition_num:
            file_name = f"transition_num{expert_data_num}{data_name}.pkl"
        else:
            file_name = f"episode_num{expert_data_num}{data_name}.pkl"
        print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), "wb") as handle:
            pickle.dump(data, handle)

    def load_expert_data(
        self,
        algo,
        env_id,
        duplicate_expert_last_state,
        data_name="",
        expert_sub_sample_ratio=-1,
        only_use_relative_state=False,
    ):
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
        if expert_sub_sample_ratio > 0:
            expert_num = len(self.expert_states)
            idx_array = np.floor(
                (1 / expert_sub_sample_ratio)
                * np.array(range(int(expert_num // (1 / expert_sub_sample_ratio))))
            ).astype(int)
            #true for idx in idx array elsa false
            sub_sample_mask = np.isin(np.array(range(expert_num)), idx_array)
        else:
            sub_sample_mask = np.ones(len(self.expert_states), dtype=bool)
        self.expert_states = self.expert_states[sub_sample_mask]
        self.expert_actions = data["actions"][sub_sample_mask]
        self.expert_rewards = data["rewards"][sub_sample_mask]
        self.expert_next_states = data["next_states"][sub_sample_mask]
        self.expert_dones = data["dones"][sub_sample_mask]
        expert_data_num = np.sum(sub_sample_mask)
        if only_use_relative_state:
            self.expert_states = np.delete(self.expert_states, np.s_[29:35], axis=1)
            self.expert_next_states = np.delete(
                self.expert_next_states, np.s_[29:35], axis=1
            )
        print(f"expert data size : {expert_data_num}")
        if self.to_tensor:
            self.expert_states = torch.FloatTensor(self.expert_states)
            self.expert_actions = torch.FloatTensor(self.expert_actions)
            self.expert_rewards = torch.FloatTensor(self.expert_rewards)
            self.expert_next_states = torch.FloatTensor(self.expert_next_states)
            self.expert_dones = torch.FloatTensor(self.expert_dones)
        if "env_states" in data.keys():
            self.expert_env_states = np.array(data["env_states"],dtype=object)[sub_sample_mask]
        if "options" in data.keys():
            self.env_reset_options = data["options"]
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
        return expert_data_num

    def __len__(self):
        return min(self.storage_index, self.size)
