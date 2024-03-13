import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class base_dqn:
    def __init__(
        self,
        observation_dim,
        action_num,
        hidden_dim=256,
        network_type="dueling",
        noisy_network=False,
        soft_update_target=True,
        gamma=0.99,
        optim="adam",
        lr=3e-4,
        tau=0.01,
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        batch_size=256,
    ):
        self.observation_dim = observation_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.noisy_network = noisy_network
        """actor"""
        self.q_network = self.get_new_network(network_type)  # vanilla, dueling

        if optim == "adam":
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        else:
            raise TypeError(f"optimizer type : {optim} not supported")

        self.best_q_network = copy.deepcopy(self.q_network)
        self.best_optimizer = copy.deepcopy(self.optimizer)

        self.q_network_target = copy.deepcopy(self.q_network)

        self.previous_checkpoint_path = None
        self.update_target = (
            self.soft_update_target if soft_update_target else self.hard_update_target
        )
        self.train()

    def train(self):
        self.q_network.train()
        self.q_network_target.train()

    def eval(self):
        self.q_network.eval()
        self.q_network_target.eval()

    def get_new_network(self, network_type):
        if network_type == "vanilla":
            return VanillaDQN(
                self.observation_dim,
                self.hidden_dim,
                self.action_num,
                self.noisy_network,
            ).to(device)
        else:
            return DuelingDQN(
                self.observation_dim,
                self.hidden_dim,
                self.action_num,
                self.noisy_network,
            ).to(device)

    def soft_update_target(self):
        for i, j in zip(
            self.q_network_target.parameters(), self.q_network.parameters()
        ):
            i.data = (1 - self.tau) * i.data + self.tau * j.data

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def hard_update_target(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def cache_weight(self):
        self.best_q_network = copy.deepcopy(self.q_network)
        self.best_optimizer = copy.deepcopy(self.optimizer)

    def save_weight(self, best_testing_reward, algo, env_id, episodes):
        path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "episodes": episodes,
            "dqn_state_dict": self.best_q_network.state_dict(),
            "dqn_optimizer_state_dict": self.best_optimizer.state_dict(),
            "reward": best_testing_reward,
        }

        file_path = os.path.join(
            path, f"episode{episodes}_reward{round(best_testing_reward,3)}.pt"
        )
        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo=None, env_id=None, path=None):
        if path is None:
            assert algo is not None and env_id is not None
            path = f"./trained_model/{algo}/{env_id}/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[0]
        else:
            assert os.path.isfile(path)

        checkpoint = torch.load(path)

        self.q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.best_q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.optimizer.load_state_dict(checkpoint["dqn_optimizer_state_dict"])
        self.best_optimizer.load_state_dict(checkpoint["dqn_optimizer_state_dict"])
        print("loaded weight from", path)

    def act(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class VanillaDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, noisy_network=False):
        super().__init__()
        self.noisy_network = noisy_network
        if self.noisy_network:
            linear_layer = noisy_linear
        else:
            linear_layer = nn.Linear
        self.fc1 = linear_layer(input_dim, hidden_dim)
        self.fc2 = linear_layer(hidden_dim, hidden_dim)
        self.fc3 = linear_layer(hidden_dim, hidden_dim)
        self.fc4 = linear_layer(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_noise(self):
        if self.noisy_network:
            self.fc1.reset_noise()
            self.fc2.reset_noise()
            self.fc3.reset_noise()
            self.fc4.reset_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, noisy_network=False):
        super().__init__()
        self.noisy_network = noisy_network
        if self.noisy_network:
            linear_layer = noisy_linear
        else:
            linear_layer = nn.Linear
        self.fc1 = linear_layer(input_dim, hidden_dim)
        self.fc2 = linear_layer(hidden_dim, hidden_dim)
        self.fc3 = linear_layer(hidden_dim, hidden_dim)
        self.fc4_value = linear_layer(hidden_dim, 1)
        self.fc4_advantage = linear_layer(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_v = self.fc4_value(x)
        x_a = self.fc4_advantage(x)
        return x_v + (x_a - torch.mean(x_a, axis=1, keepdim=True))

    def reset_noise(self):
        if self.noisy_network:
            self.fc1.reset_noise()
            self.fc2.reset_noise()
            self.fc3.reset_noise()
            self.fc4_value.reset_noise()
            self.fc4_advantage.reset_noise()


class noisy_linear(nn.Module):
    def __init__(self, input_dim, output_dim, std_zero=0.4):
        super(noisy_linear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_zero

        self.weight_mu = nn.Parameter(
            torch.FloatTensor(self.output_dim, self.input_dim)
        )
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(self.output_dim, self.input_dim)
        )
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(self.output_dim, self.input_dim)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer("bias_epsilon", torch.FloatTensor(self.output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                weight_noise = self.weight_sigma.mul(self.weight_epsilon)
                bias_noise = self.bias_sigma.mul(self.bias_epsilon)
            weight = self.weight_mu + weight_noise
            bias = self.bias_mu + bias_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
