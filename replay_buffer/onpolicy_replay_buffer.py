import numpy as np
import os
import pickle
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class onpolicy_replay_buffer:
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.storage_index = 0
        self.states = np.empty((size, state_dim))
        self.actions = np.empty((size, action_dim))
        self.rewards = np.empty((size, 1))
        self.values = np.empty((size, 1))
        self.next_values = np.empty((size, 1))
        self.log_probs = np.empty((size, 1))
        self.dones = np.empty((size, 1))

    def store(self, s, a, r, v, nv, lp, d):
        index = self.storage_index % self.size
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.values[index] = v
        self.next_values[index] = nv
        self.log_probs[index] = lp
        self.dones[index] = d
        self.storage_index += 1

    def calculate_gae(self, gamma=0.99, decay=0.97):
        """
        Return the General Advantage Estimates from the given rewards and values.
        Paper: https://arxiv.org/pdf/1506.02438.pdf
        """
        delta = []
        for i in range(self.storage_index):
            delta.append(
                self.rewards[i]
                + gamma * self.next_values[i] * (1 - self.dones[i])
                - self.values[i]
            )
        # delta = np.array(delta)
        # delta = (delta - delta.mean()) / (delta.std() + 1e-8)
        gae = [delta[-1]]
        for i in reversed(range(len(delta) - 1)):
            gae.append(delta[i] + decay * gamma * gae[-1] * (1 - self.dones[i]))

        self.gae = np.array(gae[::-1])

    def discount_rewards(self, gamma=0.99, critic=None):
        """
        Return discounted rewards based on the given rewards and gamma param.
        """
        if self.dones[self.storage_index - 1] == True:
            new_rewards = [float(self.rewards[self.storage_index - 1])]
        else:
            # if current episode is not done, use critic to estimate the value of current state
            boostrapped_value = critic(
                torch.FloatTensor(self.states[self.storage_index - 1])
                .unsqueeze(0)
                .to(device)
            ).item()
            new_rewards = [boostrapped_value]

        for i in reversed(range(self.storage_index - 1)):
            new_rewards.append(float(self.rewards[i]) + gamma * new_rewards[-1])
        self.returns = np.array(new_rewards[::-1])[..., None]

    def clear(self):
        self.storage_index = 0

    def sample(self, batch_size, expert=False):
        if expert:
            if batch_size == -1:
                indices = np.random.permutation(len(self.expert_states))
            else:
                indices = np.random.choice(len(self.expert_states), batch_size)
            return (
                self.expert_states[indices],
                self.expert_actions[indices],
                self.expert_rewards[indices],
                self.expert_next_states[indices],
                self.expert_dones[indices],
            )
        else:

            if batch_size == -1:
                indices = np.random.permutation(self.storage_index)
            else:
                indices = np.random.randint(
                    min(self.storage_index, self.size), size=batch_size
                )
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.values[indices],
                self.log_probs[indices],
                self.gae[indices],
                self.returns[indices],
            )

    def write_storage(self, based_on_transition_num, expert_data_num, algo, env_id):
        path = f"./saved_expert_transition/{env_id}/{algo}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        save_idx = min(self.storage_index, self.size)
        print(save_idx)
        data = {
            "states": self.states[:save_idx],
            "actions": self.actions[:save_idx],
            "rewards": self.rewards[:save_idx],
            "value": self.values[:save_idx],
            "log_prob": self.log_probs[:save_idx],
        }
        if based_on_transition_num:
            file_name = f"transition_num{expert_data_num}.pkl"
        else:
            file_name = f"episode_num{expert_data_num}.pkl"
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
        return min(self.storage_index, self.size)
