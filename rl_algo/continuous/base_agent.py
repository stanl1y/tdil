import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class base_agent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        action_lower=-1,
        action_upper=1,
        critic_num=1,
        hidden_dim=256,
        policy_type="stochastic",
        actor_target=True,
        critic_target=True,
        gamma=0.99,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.01,
        batch_size=256,
        state_only_critic=False,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.critic_num = critic_num
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.critic_criterion = nn.MSELoss()
        self.action_lower = action_lower
        self.action_upper = action_upper
        self.action_scale = (action_upper - action_lower) / 2
        self.action_bias = (action_upper + action_lower) / 2

        """actor"""
        self.actor = self.get_new_actor(policy_type)

        if actor_optim == "adam":
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr
            )
        else:
            raise TypeError(f"optimizer type : {critic_optim} not supported")

        self.best_actor = copy.deepcopy(self.actor)
        self.best_actor_optimizer = copy.deepcopy(self.actor_optimizer)

        if actor_target:
            self.actor_target = copy.deepcopy(self.actor)
        else:
            self.actor_target = None

        """critic"""
        self.critic = [
            self.get_new_critic(state_only=state_only_critic)
            for _ in range(self.critic_num)
        ]
        if critic_optim == "adam":
            self.critic_optimizer = [
                torch.optim.Adam(model.parameters(), lr=critic_lr)
                for model in self.critic
            ]
        else:
            raise TypeError(f"optimizer type : {critic_optim} not supported")

        if self.critic_num == 1:
            self.critic = self.critic[0]
            self.critic_optimizer = self.critic_optimizer[0]
        if critic_target:
            self.critic_target = copy.deepcopy(self.critic)
        else:
            self.critic_target = None

        self.best_critic = copy.deepcopy(self.critic)
        self.best_critic_optimizer = copy.deepcopy(self.critic_optimizer)
        self.previous_checkpoint_path = None
        self.train()

    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def get_new_actor(self, policy_type):
        if policy_type == "stochastic":
            return StochasticPolicyNet(
                self.observation_dim,
                self.hidden_dim,
                self.action_dim,
                self.action_scale,
                self.action_bias,
            ).to(device)
        elif policy_type == "fix_std_stochastic":
            return FixStdStochasticPolicyNet(
                self.observation_dim,
                self.hidden_dim,
                self.action_dim,
                self.action_scale,
                self.action_bias,
            ).to(device)
        elif policy_type == "primitive":
            return PrimitivePolicyNet(
                self.observation_dim,
                self.hidden_dim,
                self.action_dim,
            ).to(device)
        else:
            return DeterministicPolicyNet(
                self.observation_dim,
                self.hidden_dim,
                self.action_dim,
                self.action_scale,
                self.action_bias,
            ).to(device)

    def get_new_critic(self, state_only=False):
        if state_only:
            return ValueNet(self.observation_dim, self.hidden_dim, 1).to(device)
        else:
            return CriticNet(
                self.observation_dim + self.action_dim, self.hidden_dim, 1
            ).to(device)

    def soft_update_target(self):
        if self.actor_target is not None:
            for i, j in zip(self.actor_target.parameters(), self.actor.parameters()):
                i.data = (1 - self.tau) * i.data + self.tau * j.data

        if self.critic_target is not None:
            if self.critic_num > 1:
                for idx in range(self.critic_num):
                    for i, j in zip(
                        self.critic_target[idx].parameters(),
                        self.critic[idx].parameters(),
                    ):
                        i.data = (1 - self.tau) * i.data + self.tau * j.data
            else:
                for i, j in zip(
                    self.critic_target.parameters(), self.critic.parameters()
                ):
                    i.data = (1 - self.tau) * i.data + self.tau * j.data

    def hard_update_target(self):
        if self.actor_target is not None:
            self.actor_target.load_state_dict(self.actor.state_dict())

        if self.critic_target is not None:
            for idx in range(self.critic_num):
                self.critic_target[idx].load_state_dict(self.critic[idx].state_dict())

    def cache_weight(self):
        self.best_actor = copy.deepcopy(self.actor)
        self.best_actor_optimizer = copy.deepcopy(self.actor_optimizer)
        self.best_critic = copy.deepcopy(self.critic)
        self.best_critic_optimizer = copy.deepcopy(self.critic_optimizer)

    def save_weight(self, best_testing_reward, algo, env_id, episodes):
        path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "episodes": episodes,
            "actor_state_dict": self.best_actor.state_dict(),
            "actor_optimizer_state_dict": self.best_actor_optimizer.state_dict(),
            "reward": best_testing_reward,
        }
        if self.critic_num == 1:
            data["critic_state_dict"] = self.best_critic.state_dict()
            data[
                "critic_optimizer_state_dict"
            ] = self.best_critic_optimizer.state_dict()
        else:
            for idx, (model, optimizer) in enumerate(
                zip(self.best_critic, self.best_critic_optimizer)
            ):
                data[f"critic_state_dict{idx}"] = model.state_dict()
                data[f"critic_optimizer_state_dict{idx}"] = optimizer.state_dict()

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

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.best_actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.best_actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

        if self.critic_num == 1:
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.best_critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            self.best_critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
        else:
            for idx in range(self.critic_num):
                self.critic[idx].load_state_dict(checkpoint[f"critic_state_dict{idx}"])
                self.critic_target[idx].load_state_dict(
                    checkpoint[f"critic_state_dict{idx}"]
                )
                self.best_critic[idx].load_state_dict(
                    checkpoint[f"critic_state_dict{idx}"]
                )
                self.critic_optimizer[idx].load_state_dict(
                    checkpoint[f"critic_optimizer_state_dict{idx}"]
                )
                self.best_critic_optimizer[idx].load_state_dict(
                    checkpoint[f"critic_optimizer_state_dict{idx}"]
                )

    def act(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class PrimitivePolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DeterministicPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Tanh()
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.output_activation(x)
        x = x * self.action_scale + self.action_bias
        return x


class StochasticPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_scale, action_bias):
        super().__init__()
        self.max_logstd = 2
        self.min_logstd = -20
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logstd = nn.Linear(hidden_dim, output_dim)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, min=self.min_logstd, max=self.max_logstd)
        # dist = torch.distributions.normal.Normal(mu, std)
        return mu, logstd

    def sample(self, x):
        mu, logstd = self.forward(x)
        std = logstd.exp()
        dist = torch.distributions.normal.Normal(mu, std)
        x = dist.rsample()
        x_norm = torch.tanh(x)
        action = x_norm * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - x_norm.pow(2)) + 1e-8)
        log_prob = log_prob.sum(-1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu

    def get_log_prob(self, x, action):
        mu, logstd = self.forward(x)
        std = logstd.exp()
        dist = torch.distributions.normal.Normal(mu, std)
        action = (action - self.action_bias) / self.action_scale
        action = torch.atanh(torch.clip(action, max=0.999, min=-0.999))
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob


class FixStdStochasticPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)
        return mu

    def sample(self, x, std=0.1):
        mu = self.forward(x)
        dist = torch.distributions.normal.Normal(mu, std)
        x = dist.rsample()
        x_norm = torch.tanh(x)
        action = x_norm * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x)
        log_prob = log_prob.sum(-1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu

    def get_log_prob(self, x, action, std=0.1):
        mu = self.forward(x)
        dist = torch.distributions.normal.Normal(mu, std)
        action = (action - self.action_bias) / self.action_scale
        action = torch.atanh(torch.clip(action, max=0.999, min=-0.999))
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob


class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, s, a):
        s = s.float()
        a = a.float()
        x = torch.cat((s, a), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
