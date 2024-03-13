import wandb
import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import imageio
import time
import copy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class neighborhood_il_hybrid:
    def __init__(self, config):
        """get neighbor model config"""
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        self.batch_size = config.batch_size
        self.neighbor_model_alpha = config.neighbor_model_alpha
        self.neighbor_criteria = nn.BCELoss(reduction="none")
        self.ood = config.ood
        self.bc_only = config.bc_only
        self.no_bc = config.no_bc
        self.update_neighbor_frequency = config.update_neighbor_frequency
        self.update_neighbor_step = config.update_neighbor_step
        self.update_neighbor_until = config.update_neighbor_until
        self.oracle_neighbor = config.oracle_neighbor
        self.discretize_reward = config.discretize_reward
        self.log_name = config.log_name
        self.duplicate_expert_last_state = config.duplicate_expert_last_state
        self.data_name = config.data_name
        self.auto_threshold_ratio = config.auto_threshold_ratio
        self.threshold_discount_factor = config.threshold_discount_factor
        self.fix_env_random_seed = config.fix_env_random_seed
        self.render = config.render
        self.hard_negative_sampling = not config.no_hard_negative_sampling
        self.explore_step = config.explore_step
        self.use_env_done = config.use_env_done
        self.use_target_neighbor = config.use_target_neighbor
        self.tau = config.tau
        self.no_update_alpha = config.no_update_alpha
        self.total_steps = 0
        if self.hard_negative_sampling:
            print("hard negative sampling")
        if self.auto_threshold_ratio:
            self.policy_threshold_ratio = 0.1
        else:
            self.policy_threshold_ratio = config.policy_threshold_ratio

        try:
            self.margin_value = config.margin_value
        except:
            self.margin_value = 0.1
        wandb.init(
            project="Neighborhood",
            name=f"hybrid_{self.env_id}{self.log_name}_explore{self.explore_step}",
            config=config,
        )

    def gen_reward_calc_expert_data(self, storage):
        """
        generate expert next state data for reward calculation
        """
        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            -1, expert=True
        )
        if storage.to_tensor:
            self.expert_ns_data = torch.tile(
                expert_next_state, (self.batch_size, 1)
            ).to(device)
        else:
            self.expert_ns_data = np.tile(expert_next_state, (self.batch_size, 1))
            self.expert_ns_data = torch.FloatTensor(self.expert_ns_data).to(device)

        """
        generate label and weight for neighbor model training
        """
        self.update_neighor_label = (
            torch.FloatTensor(
                np.concatenate(
                    (
                        np.ones(self.batch_size),
                        np.zeros(
                            self.batch_size * (2 if self.hard_negative_sampling else 1)
                        ),
                    )
                )
            )
            .view((-1, 1))
            .to(device)
        )
        self.neighbor_loss_weight = (
            torch.cat(
                (
                    torch.ones(self.batch_size) * self.neighbor_model_alpha,
                    torch.ones(
                        self.batch_size * (2 if self.hard_negative_sampling else 1)
                    )
                    * (1 - self.neighbor_model_alpha),
                )
            )
            .view((-1, 1))
            .to(device)
        )

    def start(self, agent, env, storage, util_dict):
        if self.oracle_neighbor:
            self.NeighborhoodNet = util_dict["OracleNeighborhoodNet"].to(device)
        else:
            self.NeighborhoodNet = util_dict["NeighborhoodNet"].to(device)
            self.NeighborhoodNet_optimizer = torch.optim.Adam(
                self.NeighborhoodNet.parameters(), lr=3e-4
            )
        if self.use_target_neighbor:
            self.TargetNeighborhoodNet = copy.deepcopy(self.NeighborhoodNet).to(device)
        storage.load_expert_data(
            algo=self.algo,
            env_id=self.env_id,
            duplicate_expert_last_state=self.duplicate_expert_last_state,
            data_name=self.data_name,
        )
        self.gen_reward_calc_expert_data(storage)
        self.train(agent, env, storage)

    def test(self, agent, env, render_id=0):
        # agent.eval()
        total_reward = 0
        render = self.render and render_id % 40 == 0
        if render:
            frame_buffer = []
            if not os.path.exists(f"./experiment_logs/{self.env_id}/{self.log_name}/"):
                os.makedirs(f"./experiment_logs/{self.env_id}/{self.log_name}/")
        for i in range(3):
            state = env.reset()
            done = False
            if self.ood:
                for _ in range(5):
                    state, reward, done, info = env.step(
                        env.action_space.sample()
                    )  # env.action_space.sample()
            while not done:
                action = agent.act(state, testing=False)
                # agent.q_network.reset_noise()
                next_state, reward, done, info = env.step(action)
                if render:
                    frame_buffer.append(env.render(mode="rgb_array"))
                total_reward += reward
                state = next_state
        if render:
            imageio.mimsave(
                f"./experiment_logs/{self.env_id}/{self.log_name}/{render_id}.gif",
                frame_buffer,
            )
        total_reward /= 3
        # agent.train()
        return total_reward

    def train(self, agent, env, storage):
        if self.buffer_warmup:
            state = env.reset()
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                    done = False
                else:
                    state = next_state
        best_testing_reward = -1e7
        best_episode = 0
        print("warmup finished")
        for episode in range(self.episodes):
            if (
                not self.oracle_neighbor
                and episode % self.update_neighbor_frequency == 0
                and episode <= self.update_neighbor_until
                and not self.use_target_neighbor
            ):
                for _ in range(self.update_neighbor_step):
                    neighbor_loss = self.update_neighbor_model(storage)
                wandb.log({"neighbor_model_loss": neighbor_loss}, commit=False)
            if self.fix_env_random_seed:
                state = env.reset(seed=0)
            else:
                state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                state = next_state
                if self.use_target_neighbor:
                    neighbor_loss = self.update_neighbor_model(storage)
                loss_info = agent.update_using_neighborhood_reward(
                    storage,
                    self.NeighborhoodNet
                    if not self.use_target_neighbor
                    else self.TargetNeighborhoodNet,
                    self.expert_ns_data,
                    self.margin_value,
                    self.bc_only,
                    self.no_bc,
                    self.oracle_neighbor,
                    self.discretize_reward,
                    self.policy_threshold_ratio,
                    self.use_env_done,
                    self.no_update_alpha,
                )
                self.total_steps += 1
            for _ in range(5):
                obs, _, _, _, done, expert_env_state, idx = storage.sample(
                    batch_size=1,
                    expert=True,
                    return_idx=True,
                    return_expert_env_states=True,
                    exclude_tail_num=self.explore_step - 1,
                )
                obs = obs[0]
                env.reset()
                env.sim.set_state(expert_env_state)
                env.sim.forward()
                for _ in range(self.explore_step):
                    action = agent.act(obs)
                    next_obs, reward, done, info = env.step(action)
                    storage.store(obs, action, reward, next_obs, done)
                    obs = next_obs
                    if self.use_target_neighbor:
                        neighbor_loss = self.update_neighbor_model(storage)
                    loss_info = agent.update_using_neighborhood_reward(
                        storage,
                        self.NeighborhoodNet
                        if not self.use_target_neighbor
                        else self.TargetNeighborhoodNet,
                        self.expert_ns_data,
                        self.margin_value,
                        self.bc_only,
                        self.no_bc,
                        self.oracle_neighbor,
                        self.discretize_reward,
                        self.policy_threshold_ratio,
                        self.use_env_done,
                        self.no_update_alpha,
                    )
                    self.total_steps += 1
                    if done:
                        break

            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": episode,
                    "buffer_size": len(storage),
                    "threshold_ratio": self.policy_threshold_ratio,
                    **loss_info,
                    "neighbor_model_loss": neighbor_loss,
                    "total_steps": self.total_steps,
                }
            )
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()

            if episode % 5 == 0:
                testing_reward = self.test(
                    agent, env, render_id=episode if self.render else None
                )
                if testing_reward > best_testing_reward:
                    agent.cache_weight()
                    best_testing_reward = testing_reward
                    best_episode = episode
                wandb.log(
                    {
                        "testing_reward": testing_reward,
                        "testing_episode_num": episode,
                        "best_testing_reward": best_testing_reward,
                    }
                )
                if hasattr(env, "eval_toy_q"):
                    env.eval_toy_q(
                        agent,
                        self.NeighborhoodNet,
                        storage,
                        f"./experiment_logs/{self.env_id}{self.log_name}/",
                        episode,
                        self.oracle_neighbor,
                    )
            if episode % self.save_weight_period == 0 and not self.oracle_neighbor:
                agent.save_weight(
                    best_testing_reward, "neighborhood_il", self.env_id, best_episode
                )
                path = f"./trained_model/neighborhood/{self.env_id}/"
                if not os.path.isdir(path):
                    os.makedirs(path)
                data = {
                    "episodes": episode,
                    "neighborhood_state_dict": self.NeighborhoodNet.state_dict(),
                    "neighborhood_optimizer_state_dict": self.NeighborhoodNet_optimizer.state_dict(),
                }

                file_path = os.path.join(path, f"hybrid_episode{episode}_{self.log_name}.pt")
                torch.save(data, file_path)
                try:
                    os.remove(self.previous_checkpoint_path)
                except:
                    pass
                self.previous_checkpoint_path = file_path
            if self.auto_threshold_ratio:
                self.policy_threshold_ratio *= self.threshold_discount_factor

    def update_neighbor_model(self, storage):
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
        next_state_shift = torch.roll(next_state, 1, 0)  # for negative samples
        """
        TODO:easy positive samples(itself)
        """
        posivite_data = torch.cat((state, next_state), axis=1)
        negative_data = torch.cat((state, next_state_shift), axis=1)
        if not self.hard_negative_sampling:
            input_data = torch.cat((posivite_data, negative_data), axis=0)
        else:
            negative_hard_data = torch.cat((next_state, state), axis=1)
            input_data = torch.cat(
                (posivite_data, negative_data, negative_hard_data), axis=0
            )
        input_data = input_data.to(device)
        prediction = self.NeighborhoodNet(input_data)

        # predict positive samples
        loss = self.neighbor_criteria(
            prediction,
            self.update_neighor_label,
        )
        loss = torch.mean(loss * self.neighbor_loss_weight)
        self.NeighborhoodNet_optimizer.zero_grad()
        loss.backward()
        self.NeighborhoodNet_optimizer.step()
        if self.use_target_neighbor:
            self.update_target_neighbor_model()
        return loss.item()

    def update_target_neighbor_model(self):
        for target_param, param in zip(
            self.TargetNeighborhoodNet.parameters(), self.NeighborhoodNet.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


# CUDA_VISIBLE_DEVICES=0 python3 main.py --main_stage neighborhood_il --main_task neighborhood_dsac --env LunarLander-v2 --wrapper basic --episode 2000 --log_name expectile99_ac_duplicateLast_nextStateReward_disReward
