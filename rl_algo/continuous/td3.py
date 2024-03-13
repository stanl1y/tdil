from .base_agent import base_agent
from exploration.ounoise import OUNoise
import torch.nn as nn
import copy
import torch
import os
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class td3(base_agent):
    def __init__(
        self,
        observation_dim,
        action_dim,
        action_lower=-1,
        action_upper=1,
        hidden_dim=256,
        gamma=0.99,
        actor_optim="adam",
        critic_optim="adam",
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.01,
        batch_size=256,
        use_ounoise=True,
        policy_update_delay=2,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_lower=action_lower,
            action_upper=action_upper,
            critic_num=2,
            hidden_dim=hidden_dim,
            policy_type="deterministic",
            actor_target=True,
            critic_target=True,
            gamma=gamma,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            batch_size=batch_size,
        )
        self.update_step = 0
        self.policy_update_delay = policy_update_delay
        self.ounoise = (
            OUNoise(
                action_dimension=action_dim,
                mu=0,
                scale=self.action_scale,
            )
            if use_ounoise
            else None
        )

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state)
        if not testing:
            if self.ounoise is not None:
                action = action[0].cpu().numpy() + self.ounoise.noise()
            else:
                action = action[0].cpu().numpy() + np.random.normal(
                    self.action_bias, self.action_scale / 2, self.action_dim
                )
            return np.clip(action, self.action_lower, self.action_upper)
        else:
            return action[0].cpu().numpy()

    def update_critic(self, state, action, reward, next_state, done):

        """compute target value"""
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_action = next_action + torch.clip(
                torch.normal(0, self.action_scale / 4, size=next_action.shape),
                min=-self.action_scale / 2,
                max=self.action_scale / 2,
            ).to(device)
            next_action = torch.clip(
                next_action, min=self.action_lower, max=self.action_upper
            )
            target_q_val = [
                critic_target(next_state, next_action)
                for critic_target in self.critic_target
            ]
            target_value = reward + self.gamma * (1 - done) * (
                torch.min(target_q_val[0], target_q_val[1])
            )

        """compute loss and update"""
        q_val = [critic(state, action) for critic in self.critic]
        critic_loss = [self.critic_criterion(pred, target_value) for pred in q_val]

        for i in range(2):
            self.critic_optimizer[i].zero_grad()
            critic_loss[i].backward()
            self.critic_optimizer[i].step()
        return {
            "critic0_loss": critic_loss[0],
            "critic1_loss": critic_loss[1],
        }

    def update_actor(self, state, reward=None, threshold=None):
        if threshold != None:
            state_idx = torch.argwhere(reward.reshape(-1) < threshold).reshape(-1)
            policy_state = state[state_idx]
        else:
            policy_state = state
        filtered = state.shape[0] - policy_state.shape[0]
        actor_loss = 0
        if policy_state.shape[0] > 0:
            action = self.actor(policy_state)
            q_val = self.critic[0](policy_state, action)
            actor_loss = -q_val[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return {"actor_loss": actor_loss, "filtered": filtered}

    def bc_update(self, expert_state, expert_action):
        # if not hasattr(self, "bc_optimizer") or not hasattr(self, "bc_criterion"):
        #     self.bc_optimizer = torch.optim.Adam(
        #         self.actor.parameters(), lr=self.actor_lr
        #     )
        if not hasattr(self, "bc_criterion"):
            self.bc_criterion = nn.MSELoss()
        action = self.actor(expert_state)
        bc_loss = self.bc_criterion(action, expert_action)

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()
        return bc_loss

    def update(self, storage):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        """update model"""
        critic_loss = self.update_critic(state, action, reward, next_state, done)
        if self.update_step % self.policy_update_delay == 0:
            actor_loss = self.update_actor(state)
            self.soft_update_target()
            self.prev_actor_loss = actor_loss
        else:
            actor_loss = self.prev_actor_loss
        self.update_step += 1
        return {**critic_loss, **actor_loss}

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
        for idx in range(self.critic_num):
            self.best_critic[idx].load_state_dict(self.critic[idx].state_dict())
            self.best_critic_optimizer[idx].load_state_dict(
                self.critic_optimizer[idx].state_dict()
            )

    def save_weight(self, best_testing_reward, algo, env_id, episodes, delete_prev_weight=True):
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
        }

        for idx, (model, optimizer) in enumerate(
            zip(self.best_critic, self.best_critic_optimizer)
        ):
            data[f"critic_state_dict{idx}"] = model.cpu().state_dict()
            data[f"critic_optimizer_state_dict{idx}"] = optimizer.state_dict()

        torch.save(data, file_path)
        if delete_prev_weight and self.previous_checkpoint_path is not None:
            try:
                os.remove(self.previous_checkpoint_path)
            except:
                pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo="td3", env_id=None, path=None):
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
        self.actor_target.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.best_actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

        for idx in range(self.critic_num):
            self.critic[idx].load_state_dict(checkpoint[f"critic_state_dict{idx}"])
            self.critic[idx] = self.critic[idx].to(device)
            self.critic_target[idx].load_state_dict(
                checkpoint[f"critic_state_dict{idx}"]
            )
            self.critic_target[idx] = self.critic_target[idx].to(device)
            self.best_critic[idx].load_state_dict(checkpoint[f"critic_state_dict{idx}"])
            self.best_critic[idx] = self.best_critic[idx].to(device)
            self.critic_optimizer[idx].load_state_dict(
                checkpoint[f"critic_optimizer_state_dict{idx}"]
            )
            self.best_critic_optimizer[idx].load_state_dict(
                checkpoint[f"critic_optimizer_state_dict{idx}"]
            )
