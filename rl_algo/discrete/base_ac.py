import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class base_ac:
    def __init__(
        self,
        observation_dim,
        action_num,
        critic_num=1,
        hidden_dim=256,
        # policy_type="stochastic",
        actor_target=True,
        critic_target=True,
        soft_update_target=True,
        gamma=0.99,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.01,
        batch_size=256,
    ):
        self.observation_dim = observation_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.critic_num = critic_num
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.critic_criterion = nn.MSELoss()
        self.update_target = (
            self.soft_update_target if soft_update_target else self.hard_update_target
        )
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        """actor"""
        self.actor = self.get_new_actor()

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
        self.critic = [self.get_new_critic() for _ in range(self.critic_num)]
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

    def get_new_actor(self):
        return PolicyNet(
            self.observation_dim,
            self.hidden_dim,
            self.action_num,
        ).to(device)

    def get_new_critic(self):
        return CriticNet(self.observation_dim, self.hidden_dim, self.action_num).to(
            device
        )

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


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def evaluate(self, state):
        logit = self.forward(state)
        action_probs = self.output_activation(logit)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        action_log_prob = torch.log(action_probs + z)
        return action.detach().cpu().numpy()[0], logit, action_probs, action_log_prob


class CriticNet(nn.Module):
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
