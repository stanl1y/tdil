from .base_agent import base_agent
from exploration.ounoise import OUNoise
import torch.nn as nn
import copy
import torch
import os
import numpy as np
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class sac(base_agent):
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
        alpha_lr=3e-4,
        tau=0.01,
        batch_size=256,
        use_ounoise=False,
        log_alpha_init=0,
        no_update_alpha=False,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_lower=action_lower,
            action_upper=action_upper,
            critic_num=2,
            hidden_dim=hidden_dim,
            policy_type="stochastic",
            actor_target=False,
            critic_target=True,
            gamma=gamma,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            tau=tau,
            batch_size=batch_size,
        )
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.ones(1).to(device) * log_alpha_init)
        self.log_alpha.requires_grad = True
        self.alpha_lr = alpha_lr
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.best_log_alpha_optimizer = copy.deepcopy(self.log_alpha_optimizer)
        self.entropy_loss_weight = 1.0
        self.no_update_alpha = no_update_alpha
        self.ounoise = (
            OUNoise(
                action_dimension=action_dim,
                mu=0,
                scale=self.action_scale,
            )
            if use_ounoise
            else None
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, mu = self.actor.sample(state)
        if not testing:
            if self.ounoise is not None:
                return action[0].cpu().numpy() + self.ounoise.noise()
            else:
                return action[0].cpu().numpy()
        else:
            return mu[0].cpu().numpy()

    def update_critic(
        self, state, action, reward, next_state, done, without_entropy=False
    ):

        """compute target value"""
        with torch.no_grad():
            next_action, next_log_prob, next_mu = self.actor.sample(next_state)
            target_q_val = [
                critic_target(next_state, next_action)
                for critic_target in self.critic_target
            ]
            target_value = reward + self.gamma * (1 - done) * (
                torch.min(target_q_val[0], target_q_val[1])
                + (0 if without_entropy else -self.alpha * next_log_prob)
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

    def update_actor(
        self,
        state,
        reward=None,
        threshold=None,
        target_entropy_weight=1.0,
    ):
        if threshold != None:
            state_idx = torch.argwhere(reward.reshape(-1) < threshold).reshape(-1)
            policy_state = state[state_idx]
        else:
            policy_state = state
        filtered = state.shape[0] - policy_state.shape[0]
        actor_loss = 0
        alpha_loss = 0
        entropy_loss = 0
        log_prob = 0
        if policy_state.shape[0] > 0:
            action, log_prob, mu = self.actor.sample(policy_state)
            q_val = [critic(policy_state, action) for critic in self.critic]
            entropy_loss = -self.alpha.detach() * log_prob
            actor_loss = (
                -(
                    torch.min(q_val[0], q_val[1])
                    + self.entropy_loss_weight * entropy_loss
                )
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if not self.no_update_alpha:
                """update alpha"""
                alpha_loss = (
                    self.alpha
                    * (-log_prob - target_entropy_weight * self.target_entropy).detach()
                ).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
            entropy_loss = entropy_loss.mean()
            log_prob = log_prob.mean()
        return {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "filtered": filtered,
            "entropy_loss": entropy_loss,
            "log_prob": log_prob,
        }

    def bc_update(self, expert_state, expert_action, use_mu=True):
        # if not hasattr(self, "bc_optimizer") or not hasattr(self, "bc_criterion"):
        #     self.bc_optimizer = torch.optim.Adam(
        #         self.actor.parameters(), lr=self.actor_lr
        #     )
        self.bc_criterion = nn.MSELoss()
        action, log_prob, mu = self.actor.sample(expert_state)
        if use_mu:
            bc_loss = self.bc_criterion(mu, expert_action)
        else:
            bc_loss = self.bc_criterion(action, expert_action)

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()
        return bc_loss

    def update(self, storage):

        """sample data"""
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        """update model"""
        critic_loss = self.update_critic(state, action, reward, next_state, done)
        actor_loss = self.update_actor(state)
        self.soft_update_target()
        return {**critic_loss, **actor_loss}
        # return {
        #     "critic0_loss": critic_loss[0],
        #     "critic1_loss": critic_loss[1],
        #     "actor_loss": actor_loss,
        #     "alpha_loss": alpha_loss,
        #     "alpha": self.alpha,
        # }

    def set_state_neighborhood_reward_cuda(
        self, NeighborhoodNet, expert_ns_data, next_state, state_idx, explore_step
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
        cartesian_product_state = torch.cat(
            (
                torch.repeat_interleave(next_state, explore_step + 2, dim=0),
                expert_ns_data[state_idx].reshape((-1, expert_ns_data.shape[-1])),
            ),
            dim=1,
        )

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
            reward = prob.reshape((len(next_state), -1)).sum(axis=1, keepdims=True)
            # reward, _ = torch.max(reward, dim=1, keepdim=True)
        return reward

    def set_state_neighborhood_reward(
        self, NeighborhoodNet, expert_ns_data, next_state, state_idx, explore_step
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
        cartesian_product_state = np.concatenate(
            (
                np.repeat(next_state, explore_step, axis=0),
                expert_ns_data[state_idx].reshape((-1, expert_ns_data.shape[-1])),
            ),
            axis=1,
        )
        cartesian_product_state = torch.FloatTensor(cartesian_product_state).to(device)

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
            reward = prob.reshape((len(next_state), -1)).sum(axis=1, keepdims=True)
            # reward, _ = torch.max(reward, dim=1, keepdim=True)
        return reward

    def update_using_set_state_neighborhood_reward(
        self,
        storage,
        NeighborhoodNet,
        expert_ns_data,
        explore_step,
        bc_only=False,
        no_bc=False,
        oracle_neighbor=False,
        policy_threshold_ratio=0.5,
        use_env_done=False,
    ):
        """sample agent data"""
        state, action, _, next_state, done, state_idx = storage.sample(
            self.batch_size, return_idx=True
        )
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)

        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        reward = self.set_state_neighborhood_reward_cuda(
            NeighborhoodNet, expert_ns_data, next_state, state_idx, explore_step
        )

        (
            expert_state,
            expert_action,
            _,
            expert_next_state,
            expert_done,
            expert_state_idx,
        ) = storage.sample(
            self.batch_size,
            expert=True,
            return_idx=True,
            exclude_tail_num=explore_step + 2,
        )
        if not storage.to_tensor:
            expert_state = torch.FloatTensor(expert_state)
            expert_action = torch.FloatTensor(expert_action)
            expert_next_state = torch.FloatTensor(expert_next_state)
            expert_done = torch.FloatTensor(expert_done)
        expert_state = expert_state.to(device)
        expert_action = expert_action.to(device)
        expert_next_state = expert_next_state.to(device)
        expert_done = expert_done.to(device)
        if not use_env_done:
            done *= 0
            expert_done *= 0
        expert_reward = self.set_state_neighborhood_reward_cuda(
            NeighborhoodNet,
            expert_ns_data,
            expert_next_state,
            expert_state_idx,
            explore_step,
        )

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
            expert_reward,  # can be changed to all ones
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
        if no_bc:
            bc_loss = 0
        else:
            bc_loss = self.bc_update(expert_state, expert_action)

        self.soft_update_target()
        return {**actor_loss, **critic_loss, **tmp, **reward_dict, "bc_loss": bc_loss}

    def neighborhood_reward_cuda(
        self,
        NeighborhoodNet,
        expert_ns_data,
        next_state,
        oracle_neighbor,
        discretize_reward=False,
        use_top_k=False,
        k_of_topk=1,
        complementary_reward=False,
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
        if complementary_reward:
            reward = reward.topk(20, dim=1, sorted=False)[0]
            reward = 1-reward
            #one minus product of top5 1-probabilities
            reward = 1 - torch.sqrt(reward.prod(axis=1, keepdims=True))
        else:
            if use_top_k:
                reward = reward.topk(k_of_topk, dim=1, sorted=False)[0]
            reward = reward.sum(axis=1, keepdims=True)
        # print("reshape time", time.time() - t)
        # print("end neighborhood reward")
        return reward

    def neighborhood_reward(
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
        cartesian_product_state = np.concatenate(
            (
                np.repeat(next_state, len(expert_ns_data) // self.batch_size, axis=0),
                expert_ns_data,
            ),
            axis=1,
        )
        # print("concatenate time", time.time() - t)
        # t = time.time()
        cartesian_product_state = torch.FloatTensor(cartesian_product_state).to(device)

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
            # print("prob time", time.time() - t)
            # t = time.time()
        if discretize_reward:
            prob[prob > 0.5] = 0.5
            prob[prob <= 0.5] = 0
        reward = prob.reshape((len(next_state), -1)).sum(axis=1, keepdims=True)
        # print("reshape time", time.time() - t)
        # print("end neighborhood reward")
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
        use_relative_reward=False,
        state_only=False,
        critic_without_entropy=False,
        target_entropy_weight=1.0,
        reward_scaling_weight=1.0,
        use_true_expert_relative_reward=False,
        use_top_k=False,
        k_of_topk=1,
        InverseDynamicModule=None,
        complementary_reward=False,
        discriminator=None,
        beta=0.1
    ):
        # print("in update_using_neighborhood_reward")
        # t = time.time()
        """sample agent data"""
        state, action, _, next_state, done = storage.sample(self.batch_size)
        # print("sample time", time.time() - t)
        # t = time.time()
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        # print("to tensor and device time", time.time() - t)
        # t = time.time()
        reward = self.neighborhood_reward_cuda(
            NeighborhoodNet,
            expert_ns_data,
            next_state,
            oracle_neighbor,
            discretize_reward,
            use_top_k=use_top_k,
            k_of_topk=k_of_topk,
            complementary_reward=complementary_reward
        )
        # print("neighborhood reward time", time.time() - t)
        # t = time.time()
        if discriminator is not None:
            with torch.no_grad():
                data=torch.cat((state, action), axis=1)
                prob = discriminator(data)
                prob = prob.reshape(reward.shape)
            reward = reward * (1 - beta) + prob * beta
            

        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            self.batch_size, expert=True
        )
        # print("expert sample time", time.time() - t)
        # t = time.time()
        if not storage.to_tensor:
            expert_state = torch.FloatTensor(expert_state)
            expert_action = torch.FloatTensor(expert_action)
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
            use_top_k=use_top_k,
            complementary_reward=complementary_reward,
        )
        # print("expert neighborhood reward time", time.time() - t)
        # t = time.time()
        if discriminator is not None:
            with torch.no_grad():
                data=torch.cat((expert_state, expert_action), axis=1)
                expert_prob = discriminator(data)
                expert_prob = expert_prob.reshape(reward.shape)
            expert_reward = expert_reward * (1 - beta) + expert_prob * beta
        if not use_env_done:
            done *= 0
            expert_done *= 0
            # try to use no done in imitation learning
        expert_reward_mean = expert_reward.mean().item()

        actor_loss = {}
        if not bc_only:
            actor_loss = self.update_actor(
                state,
                reward=reward,
                threshold=expert_reward_mean * policy_threshold_ratio,
            )
            # expert_actor_loss = self.update_actor(
            #     expert_state,
            #     no_update_alpha=no_update_alpha,
            # )
        # print("update actor time", time.time() - t)
        # t = time.time()
        if use_relative_reward:
            relative_reward = reward / (expert_reward_mean + 1e-6)
            relative_reward *= reward_scaling_weight
            if use_true_expert_relative_reward:
                relative_expert_reward = expert_reward / (expert_reward_mean + 1e-6)
                relative_expert_reward *= reward_scaling_weight
            else:
                relative_expert_reward = expert_reward_ones
            # relative_expert_reward *= reward_scaling_weight
        reward *= reward_scaling_weight
        expert_reward *= reward_scaling_weight
        critic_loss = self.update_critic(
            state,
            action,
            relative_reward if use_relative_reward else reward,
            next_state,
            done,
            without_entropy=critic_without_entropy,
        )
        # print("update critic time", time.time() - t)
        # t = time.time()
        if not state_only or InverseDynamicModule is not None:
            if InverseDynamicModule is not None:
                input_data = torch.cat((expert_state, expert_next_state), axis=1)
                expert_action = InverseDynamicModule(
                    input_data,
                ).detach()
            expert_critic_loss = self.update_critic(
                expert_state,
                expert_action,
                relative_expert_reward if use_relative_reward else expert_reward,
                expert_next_state,
                expert_done,
                without_entropy=critic_without_entropy,
            )
            expert_keys = {
                "expert_critic0_loss": "critic0_loss",
                "expert_critic1_loss": "critic1_loss",
            }
            expert_critic_loss = dict(
                (key, expert_critic_loss[expert_keys[key]])
                for key in expert_keys.keys()
            )
        else:
            expert_critic_loss = {}
        # print("update expert critic time", time.time() - t)
        # t = time.time()
        reward_dict = {
            "sampled_expert_reward_mean": expert_reward.mean().item(),
            "sampled_agent_reward_mean": reward.mean().item(),
        }
        if discriminator is not None:
            reward_dict["agent_IRL_reward_mean"]=prob.mean().item()
            reward_dict["expert_IRL_reward_mean"]=expert_prob.mean().item()
        if use_relative_reward:
            reward_dict[
                "sampled_agent_relative_reward_mean"
            ] = relative_reward.mean().item()
        if no_bc or state_only:
            bc_loss = 0
        else:
            bc_loss = self.bc_update(expert_state, expert_action)
        # print("update bc time", time.time() - t)
        # t = time.time()
        self.soft_update_target()
        # print("end update_using_neighborhood_reward")
        return {
            **actor_loss,
            **critic_loss,
            **expert_critic_loss,
            **reward_dict,
            "bc_loss": bc_loss,
        }

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

    def save_weight(
        self,
        best_testing_reward,
        algo,
        env_id,
        episodes,
        log_name="",
        delete_prev_weight=True,
        oracle_reward=True,
    ):
        dir_path = f"./trained_model/{algo}/{env_id}/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(
            dir_path,
            f"episode{episodes}_reward{round(best_testing_reward,3)}{log_name}.pt",
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
        if delete_prev_weight:
            if oracle_reward:
                try:
                    os.remove(self.previous_checkpoint_path)
                except:
                    pass
                self.previous_checkpoint_path = file_path
            else:
                try:
                    os.remove(self.neighbor_reward_previous_checkpoint_path)
                except:
                    pass
                self.neighbor_reward_previous_checkpoint_path = file_path

    def load_weight(self, algo="sac", env_id=None, path=None):
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
