from .base_ac import base_ac
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import copy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class discrete_sac(base_ac):
    def __init__(
        self,
        observation_dim,
        action_num,
        hidden_dim=256,
        gamma=0.99,
        optim="adam",
        soft_update_target=True,
        lr=3e-4,
        tau=0.01,
        batch_size=256,
        log_alpha_init=0,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_num=action_num,
            critic_num=2,
            hidden_dim=hidden_dim,
            # policy_type="stochastic",
            actor_target=True,
            critic_target=True,
            soft_update_target=soft_update_target,
            gamma=gamma,
            actor_optim=optim,
            critic_optim=optim,
            actor_lr=lr,
            critic_lr=lr,
            tau=tau,
            batch_size=batch_size,
        )
        self.alpha_lr = lr
        self.clip_grad_param = 1
        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = -np.log(1.0 / action_num) * 0.98
        self.log_alpha = nn.Parameter(torch.ones(1).to(device) * log_alpha_init)
        self.log_alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=lr)
        self.best_log_alpha_optimizer = copy.deepcopy(self.log_alpha_optimizer)

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, logit, action_probs, log_pis = self.actor.evaluate(state)
        if testing:
            return np.argmax(action_probs[0].cpu().detach().numpy())
        else:
            return action

    def calc_policy_loss(self, state):
        _, logit, action_probs, action_log_prob = self.actor.evaluate(
            state
        )  # prob and log prob of each action
        with torch.no_grad():
            q0 = self.critic[0](state)  # q value of each action in the state
            q1 = self.critic[1](state)
            # min_Q = torch.min(q0, q1)
        entropy = -torch.sum(action_probs * action_log_prob, dim=1, keepdim=True)
        q = torch.sum(torch.min(q0, q1) * action_probs, dim=1, keepdim=True)
        actor_loss = (-self.alpha * entropy - q).sum(1).mean()
        return actor_loss, entropy

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def update_ac(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        bc_only=False,
        train_expert_data=False,
        threshold=None,
    ):
        actor_loss = 0
        entropy = 0
        alpha_loss = 0
        filtered = 0
        if not bc_only and not train_expert_data:
            if threshold != None:
                state_idx = torch.argwhere(reward.reshape(-1) < threshold).reshape(-1)
                policy_state = state[state_idx]
            else:
                policy_state = state
            filtered = state.shape[0] - policy_state.shape[0]
            if policy_state.shape[0] > 0:
                actor_loss, entropy = self.calc_policy_loss(policy_state)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                """update alpha"""
                entropy = entropy.mean()
                alpha_loss = (
                    self.log_alpha
                    * (self.target_entropy - entropy).exp().detach()  # .exp()
                )
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

        """compute target value"""
        with torch.no_grad():
            (
                next_action,
                logit,
                next_action_probs,
                next_action_log_prob,
            ) = self.actor.evaluate(next_state)
            next_target_q_val = [
                critic_target(next_state) for critic_target in self.critic_target
            ]
            q_target_next = next_action_probs * (
                torch.min(next_target_q_val[0], next_target_q_val[1])
                - self.alpha.to(device) * next_action_log_prob
            )
            q_target_next = q_target_next.sum(dim=1, keepdim=True)
            # Compute Q targets for current states (y_i)

            target_value = reward + (self.gamma * (1 - done) * q_target_next)

        """compute loss and update"""

        q_val = [critic(state).gather(1, action.type(torch.int64)) for critic in self.critic]
        critic_loss = [self.critic_criterion(pred, target_value) for pred in q_val]

        # Update critics
        for i in range(2):
            self.critic_optimizer[i].zero_grad()
            critic_loss[i].backward()
            clip_grad_norm_(self.critic[i].parameters(), self.clip_grad_param)
            self.critic_optimizer[i].step()

        return {
            "critic0_loss": critic_loss[0],
            "critic1_loss": critic_loss[1],
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": entropy,
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "filtered": filtered,
        }

    # def DQfD_update(self, expert_state, expert_action, margin_value=0.1):
    #     q_val = self.q_network(expert_state)
    #     max_action = self.q_network(expert_state).argmax(1).unsqueeze(1)
    #     max_action_q_val = q_val.gather(1, max_action)
    #     expert_action_q_val = q_val.gather(1, expert_action)
    #     loss = (
    #         max_action_q_val + (max_action != expert_action) * margin_value
    #     ) - expert_action_q_val
    #     loss = loss.mean()
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss

    def bc_update(self, expert_state, expert_action):
        # if not hasattr(self, "bc_optimizer") or not hasattr(self, "bc_criterion"):
        #     self.bc_optimizer = torch.optim.Adam(
        #         self.actor.parameters(), lr=self.actor_lr
        #     )
        self.bc_criterion = nn.CrossEntropyLoss()
        _, logit, action_probs, action_log_prob = self.actor.evaluate(
            expert_state
        )  # prob and log prob of each action
        bc_loss = self.bc_criterion(logit, expert_action.view(-1))

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()
        return bc_loss

    def neighborhood_reward_cuda(
        self,
        NeighborhoodNet,
        expert_ns_data,
        next_state,
        oracle_neighbor,
        discretize_reward=False,
    ):
        """construct a tensor that looks like
        |s'^a_1, s^e_1|
        |s'^a_1, s^e_2|
        |s'^a_1, s^e_3|
        | ......
        |s'^a_m, s^e_{n-1}|
        |s'^a_m, s^e_n|

        it is in the shape (len(state)*len(expert_state), 2*observation_dim)
        """
        # print("in neighborhood reward")
        # t = time.time()
        # print("next_state to device time", time.time() - t)
        # t = time.time()
        cartesian_product_state = torch.cat(
            (
                torch.repeat_interleave(
                    next_state, len(expert_ns_data) // self.batch_size, dim=0
                ),
                expert_ns_data,
            ),
            dim=1,
        )
        # print("concatenate time", time.time() - t)
        # t = time.time()

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
            # print("prob time", time.time() - t)
            # t = time.time()
        if discretize_reward:
            prob[prob > 0.5] = 0.5
            prob[prob <= 0.5] = 0
        reward = prob.reshape((len(next_state), -1))
        reward = reward.sum(axis=1, keepdims=True)
        # print("reshape time", time.time() - t)
        # print("end neighborhood reward")
        return reward
    
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
        
        expert_next_state = np.concatenate(
                (expert_next_state, np.array([[-0.3, -0.4]])), axis=0
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
        expert_ns_data,
        expert_reward_ones,
        margin_value,
        bc_only=False,
        no_bc=False,
        oracle_neighbor=False,
        discretize_reward=False,
        policy_threshold_ratio=0.5,
        use_env_done=False,
        *argv,
    ):
        """sample agent data"""
        state, action, _, next_state, done = storage.sample(self.batch_size)

        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        reward = self.neighborhood_reward_cuda(
            NeighborhoodNet,
            expert_ns_data,
            next_state,
            oracle_neighbor,
            discretize_reward,
        )
        # reward = self.neighborhood_reward(
        #     NeighborhoodNet, storage, next_state, oracle_neighbor, discretize_reward
        # )


        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            self.batch_size, expert=True
        )

        if not storage.to_tensor:
            expert_state = torch.FloatTensor(expert_state)
            expert_action = torch.LongTensor(expert_action)
            expert_next_state = torch.FloatTensor(expert_next_state)
            expert_done = torch.FloatTensor(expert_done)
        expert_state = expert_state.to(device)
        expert_action = expert_action.to(device)
        expert_next_state = expert_next_state.to(device)
        expert_done = expert_done.to(device)

        expert_reward = self.neighborhood_reward_cuda(
            NeighborhoodNet,
            expert_ns_data,
            expert_next_state,
            oracle_neighbor,
            discretize_reward,
        )

        if not use_env_done:
            done *= 0
            expert_done *= 0

        expert_reward_mean = expert_reward.mean().item()

        ac_loss = self.update_ac(
            state,
            action,
            reward,
            next_state,
            done,
            bc_only,
            threshold=expert_reward_mean * policy_threshold_ratio,
        )
        reward_dict = {
            "expert_reward_mean": expert_reward_mean,
            "sampled_exp_reward_mean": reward.mean().item(),
        }
        bc_loss = self.bc_update(expert_state, expert_action)

        """we are not updating actor here"""
        expert_ac_loss = self.update_ac(
            expert_state,
            expert_action,
            expert_reward,
            expert_next_state,
            expert_done,
            bc_only,
            train_expert_data=True,
        )
        expert_keys = {
            "expert_critic0_loss": "critic0_loss",
            "expert_critic1_loss": "critic1_loss",
            "expert_actor_loss": "actor_loss",
        }
        tmp = dict(
            (key, expert_ac_loss[expert_keys[key]]) for key in expert_keys.keys()
        )

        self.update_target()
        return {**ac_loss, **tmp, **reward_dict, "bc_loss": bc_loss}

    def update(self, storage):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(
            device
        )  # keep action int type
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        """update model"""
        ac_loss = self.update_ac(state, action, reward, next_state, done)
        self.update_target()
        return ac_loss  # this dict will be recorded by wandb

    def cache_weight(self):
        self.best_actor.load_state_dict(self.actor.state_dict())
        self.best_actor_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        for idx in range(self.critic_num):
            self.best_critic[idx].load_state_dict(self.critic[idx].state_dict())
            self.best_critic_optimizer[idx].load_state_dict(
                self.critic_optimizer[idx].state_dict()
            )
        self.best_log_alpha = self.log_alpha
        self.best_log_alpha_optimizer.load_state_dict(
            self.log_alpha_optimizer.state_dict()
        )

    def save_weight(self, best_testing_reward, algo, env_id, episodes, **kwargs):
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
            "log_alpha_state_dict": self.best_log_alpha.cpu(),
            "log_alpha_optimizer_state_dict": self.best_log_alpha_optimizer.state_dict(),
            "reward": best_testing_reward,
        }

        for idx, (model, optimizer) in enumerate(
            zip(self.best_critic, self.best_critic_optimizer)
        ):
            data[f"critic_state_dict{idx}"] = model.cpu().state_dict()
            data[f"critic_optimizer_state_dict{idx}"] = optimizer.state_dict()

        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo="discrete_sac", env_id=None, path=None):
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

        self.log_alpha = checkpoint["log_alpha_state_dict"]
        self.log_alpha.requires_grad = False
        self.log_alpha = self.log_alpha.to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.log_alpha_optimizer.load_state_dict(
            checkpoint["log_alpha_optimizer_state_dict"]
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
