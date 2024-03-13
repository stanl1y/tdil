from wandb import agent
from .base_agent import base_agent
import torch.nn as nn
import copy
import torch
import os
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ppo(base_agent):
    def __init__(
        self,
        observation_dim,
        action_dim,
        action_lower=-1,
        action_upper=1,
        hidden_dim=256,
        gamma=0.99,
        lambda_decay=0.97,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        batch_size=-1,
        target_kl=0.015,
        ppo_clip_value=0.2,
        max_policy_train_iters=80,
        value_train_iters=80,
        horizon=2048,
        para_std=True,
        action_std=0.6,
        action_std_decay_rate=0.9,
        min_action_std=0.1,
        action_std_decay_freq=25000,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_lower=action_lower,
            action_upper=action_upper,
            critic_num=1,
            hidden_dim=hidden_dim,
            policy_type="stochastic" if para_std else "fix_std_stochastic",
            actor_target=False,
            critic_target=False,
            gamma=gamma,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            batch_size=batch_size,
            state_only_critic=True,
        )
        self.para_std = para_std
        self.action_std = action_std
        self.target_kl = target_kl
        self.ppo_clip_value = ppo_clip_value
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.lambda_decay = lambda_decay
        self.horizon = horizon
        self.action_std = action_std
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.action_std_decay_freq = action_std_decay_freq

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if self.para_std:
                action, log_prob, mu = self.actor.sample(state)
            else:
                action, log_prob, mu = self.actor.sample(state, self.action_std)

        if not testing:
            return action[0].cpu().numpy(), log_prob[0].cpu().numpy()
        else:
            return mu[0].cpu().numpy()

    def update_std(self):
        self.action_std = max(
            self.action_std * self.action_std_decay_rate, self.min_action_std
        )

    def get_state_value(self, state):
        value = (
            self.critic(torch.FloatTensor(state).unsqueeze(0).to(device))
            .detach()
            .cpu()
            .numpy()
        )
        return value

    def update_critic(self, state, returns):
        for _ in range(self.value_train_iters):
            state_val = self.critic(state)
            critic_loss = self.critic_criterion(state_val, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        return {"critic_loss": critic_loss}

    def update_actor(
        self, state, action, old_log_prob, gae, reward=None, threshold=None
    ):
        if threshold != None:
            state_idx = torch.argwhere(reward.reshape(-1) < threshold).reshape(-1)
            policy_state = state[state_idx]
        else:
            policy_state = state
        filtered = state.shape[0] - policy_state.shape[0]
        actor_loss = 0
        for _ in range(self.max_policy_train_iters):
            if policy_state.shape[0] > 0:
                if self.para_std:
                    new_log_prob = self.actor.get_log_prob(policy_state, action)
                else:
                    new_log_prob = self.actor.get_log_prob(
                        policy_state, action, self.action_std
                    )
                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * gae
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.ppo_clip_value, 1.0 + self.ppo_clip_value
                    )
                    * gae
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                kl = (old_log_prob - new_log_prob).mean()
                if kl > self.target_kl:
                    break

        return {
            "actor_loss": actor_loss,
            "filtered": filtered,
        }

    def bc_update(self, expert_state, expert_action):
        if not hasattr(self, "bc_optimizer") or not hasattr(self, "bc_criterion"):
            self.bc_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.actor_lr
            )
            self.bc_criterion = nn.MSELoss()
        if self.para_std:
            action, log_prob, mu = self.actor.sample(expert_state)
        else:
            action, log_prob, mu = self.actor.sample(expert_state, self.action_std)
        bc_loss = self.bc_criterion(mu, expert_action)

        self.bc_optimizer.zero_grad()
        bc_loss.backward()
        self.bc_optimizer.step()
        return bc_loss

    def update(self, storage):
        """sample data"""
        state, action, reward, value, log_prob, gae, returns = storage.sample(
            self.batch_size
        )
        """compute gae"""
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        log_prob = torch.FloatTensor(log_prob).to(device)
        gae = torch.FloatTensor(gae).to(device)
        returns = torch.FloatTensor(returns).to(device)

        """update model"""
        critic_loss = self.update_critic(state, returns)
        actor_loss = self.update_actor(state, action, log_prob, gae)
        return {**critic_loss, **actor_loss}
        # return {
        #     "critic0_loss": critic_loss[0],
        #     "critic1_loss": critic_loss[1],
        #     "actor_loss": actor_loss,
        #     "alpha_loss": alpha_loss,
        #     "alpha": self.alpha,
        # }

    def neighborhood_reward(
        self,
        NeighborhoodNet,
        storage,
        next_state,
        oracle_neighbor,
        discretize_reward=False,
    ):
        """sample expert data"""
        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            -1, expert=True
        )

        """construct a tensor that looks like
        |s'^a_1, s^e_1|
        |s'^a_1, s^e_2|
        |s'^a_1, s^e_3|
        | ......
        |s'^a_m, s^e_{n-1}|
        |s'^a_m, s^e_n|

        it is in the shape (len(state)*len(expert_state), 2*observation_dim)
        """

        cartesian_product_state = np.concatenate(
            (
                np.repeat(next_state, len(expert_next_state), axis=0),
                np.tile(expert_next_state, (len(next_state), 1)),
            ),
            axis=1,
        )
        cartesian_product_state = torch.FloatTensor(cartesian_product_state).to(device)

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
        if discretize_reward:
            prob[prob > 0.5] = 0.5
            prob[prob <= 0.5] = 0
        reward = prob.reshape((len(next_state), len(expert_next_state))).sum(
            axis=1, keepdims=True
        )
        return reward

    def update_using_neighborhood_reward(
        self,
        storage,
        NeighborhoodNet,
        margin_value,
        bc_only=False,
        oracle_neighbor=False,
        discretize_reward=False,
        policy_threshold_ratio=0.5,
    ):
        """sample agent data"""
        state, action, _, next_state, done = storage.sample(self.batch_size)

        reward = self.neighborhood_reward(
            NeighborhoodNet, storage, next_state, oracle_neighbor, discretize_reward
        )

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.zeros_like(done)).to(
            device
        )  # try to use no done in imitation learning
        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            self.batch_size, expert=True
        )

        expert_reward = self.neighborhood_reward(
            NeighborhoodNet,
            storage,
            expert_next_state,
            oracle_neighbor,
            discretize_reward,
        )
        expert_state = torch.FloatTensor(expert_state).to(device)
        expert_action = torch.FloatTensor(expert_action).to(device)
        expert_next_state = torch.FloatTensor(expert_next_state).to(device)
        expert_done = torch.FloatTensor(np.zeros_like(expert_done)).to(device)
        expert_reward_mean = expert_reward.mean().item()

        actor_loss = {}
        if not bc_only:
            actor_loss = self.update_actor(
                state,
                reward=reward,
                threshold=expert_reward_mean * policy_threshold_ratio,
            )
        critic_loss = self.update_critic(
            state,
            action,
            reward,
            next_state,
            done,
        )
        expert_critic_loss = self.update_critic(
            expert_state,
            expert_action,
            expert_reward,
            expert_next_state,
            expert_done,
        )
        expert_keys = {
            "expert_critic0_loss": "critic0_loss",
            "expert_critic1_loss": "critic1_loss",
        }
        tmp = dict(
            (key, expert_critic_loss[expert_keys[key]]) for key in expert_keys.keys()
        )
        reward_dict = {
            "expert_reward_mean": expert_reward_mean,
            "sampled_exp_reward_mean": reward.mean().item(),
        }
        bc_loss = self.bc_update(expert_state, expert_action)

        self.soft_update_target()
        return {**actor_loss, **critic_loss, **tmp, **reward_dict, "bc_loss": bc_loss}

    def cache_weight(self):
        self.best_actor.load_state_dict(self.actor.state_dict())
        self.best_actor_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.best_critic.load_state_dict(self.critic.state_dict())
        self.best_critic_optimizer.load_state_dict(self.critic_optimizer.state_dict())

    def save_weight(self, best_testing_reward, algo, env_id, episodes):
        dir_path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(
            dir_path, f"episode{episodes}_reward{round(best_testing_reward,3)}.pt"
        )

        if file_path == self.previous_checkpoint_path:
            return

        data = {
            "episodes": episodes,
            "actor_state_dict": self.best_actor.cpu().state_dict(),
            "actor_optimizer_state_dict": self.best_actor_optimizer.state_dict(),
            "reward": best_testing_reward,
            "critic_state_dict": self.best_critic.cpu().state_dict(),
            "critic_optimizer_state_dict": self.best_critic_optimizer.state_dict(),
        }

        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo="ppo", env_id=None, path=None):
        if path is None:
            assert env_id is not None
            path = f"./trained_model/{algo}/{env_id}/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[-1]
        else:
            assert os.path.isfile(path)

        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor = self.actor.to(device)
        self.best_actor.load_state_dict(checkpoint["actor_state_dict"])
        self.best_actor = self.best_actor.to(device)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.best_actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

        self.critic.load_state_dict(checkpoint[f"critic_state_dict"])
        self.critic = self.critic.to(device)
        self.best_critic.load_state_dict(checkpoint[f"critic_state_dict"])
        self.best_critic = self.best_critic.to(device)
        self.critic_optimizer.load_state_dict(
            checkpoint[f"critic_optimizer_state_dict"]
        )
        self.best_critic_optimizer.load_state_dict(
            checkpoint[f"critic_optimizer_state_dict"]
        )
