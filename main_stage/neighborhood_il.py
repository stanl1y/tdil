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


class neighborhood_il:
    def __init__(self, config):
        """get neighbor model config"""
        self.episodes = config.episodes
        self.total_timesteps=config.total_timesteps
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
        self.use_env_done = config.use_env_done
        self.use_target_neighbor = config.use_target_neighbor
        self.tau = config.tau
        self.neighborhood_tau = config.neighborhood_tau
        self.entropy_loss_weight_decay_rate = config.entropy_loss_weight_decay_rate
        self.infinite_neighbor_buffer = config.infinite_neighbor_buffer
        self.bc_pretraining = config.bc_pretraining
        self.hybrid = config.hybrid
        self.use_relative_reward = config.use_relative_reward
        self.state_only = config.state_only
        self.critic_without_entropy = config.critic_without_entropy
        self.target_entropy_weight = config.target_entropy_weight
        self.reward_scaling_weight = config.reward_scaling_weight
        self.use_true_expert_relative_reward = config.use_true_expert_relative_reward
        self.low_hard_negative_weight = config.low_hard_negative_weight
        self.use_top_k = config.use_top_k
        self.k_of_topk = config.k_of_topk
        self.use_pretrained_neighbor = config.use_pretrained_neighbor
        self.pretrained_neighbor_weight_path = config.pretrained_neighbor_weight_path
        self.expert_sub_sample_ratio = config.expert_sub_sample_ratio
        self.use_IDM = config.use_IDM
        self.reset_as_expert_state = config.reset_as_expert_state
        self.initial_state_key_to_add_noise = config.initial_state_key_to_add_noise
        self.initial_state_noise_std = config.initial_state_noise_std
        self.complementary_reward = config.complementary_reward
        self.only_use_relative_state = config.only_use_relative_state
        self.use_discriminator = config.use_discriminator
        self.beta=config.beta
        if self.env_id in [
            "AdroitHandDoor-v1",
            "AdroitHandHammer-v1",
            "AdroitHandPen-v1",
            "AdroitHandRelocate-v1",
        ]:
            self.record_success_rate = True
        else:
            self.record_success_rate = False
        if self.hard_negative_sampling:
            print("hard negative sampling")
        if self.auto_threshold_ratio:
            self.policy_threshold_ratio = 0.1
        else:
            self.policy_threshold_ratio = config.policy_threshold_ratio
            print(f"policy_threshold_ratio: {self.policy_threshold_ratio}")

        try:
            self.margin_value = config.margin_value
        except:
            self.margin_value = 0.1
        wandb.init(
            project="Neighborhood",
            name=f"{self.env_id}{self.log_name}",
            config=config,
        )

    def gen_data(self, storage):
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
            self.testing_expert_ns_data = (
                torch.tile(expert_next_state, (1000, 1)).to(device).to(device)
            )
            self.testing_expert_ns_data0 = torch.repeat_interleave(
                expert_next_state, self.expert_data_num, dim=0
            ).to(device)
        else:
            # add [0.3,0.4] into expert_next_state, dtype=np.array
            # expert_next_state = np.concatenate(
            #     (expert_next_state, np.array([[-0.3, -0.4]])), axis=0
            # )
            self.expert_ns_data = np.tile(expert_next_state, (self.batch_size, 1))
            self.expert_ns_data = torch.FloatTensor(self.expert_ns_data).to(device)
            self.testing_expert_ns_data = np.tile(expert_next_state, (1000, 1))
            self.testing_expert_ns_data = torch.FloatTensor(
                self.testing_expert_ns_data
            ).to(device)
            self.testing_expert_ns_data0 = torch.FloatTensor(
                np.repeat(expert_next_state, self.expert_data_num, axis=0)
            ).to(device)
        self.expert_cartesian_product_state = torch.cat(
            (
                self.testing_expert_ns_data0,
                # in mujoco, 1000 is the number of expert states
                self.testing_expert_ns_data[
                    : len(self.testing_expert_ns_data0)
                ].reshape((-1, self.testing_expert_ns_data.shape[-1])),
            ),
            dim=1,
        )

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
        if self.hard_negative_sampling:

            self.neighbor_loss_weight = (
                torch.cat(
                    (
                        torch.ones(self.batch_size) * self.neighbor_model_alpha,
                        torch.ones(self.batch_size) * (1 - self.neighbor_model_alpha),
                        torch.ones(self.batch_size)
                        * (
                            (1 - self.neighbor_model_alpha)
                            if self.low_hard_negative_weight
                            else 1.0
                        ),
                    )
                )
                .view((-1, 1))
                .to(device)
            )
        else:
            self.neighbor_loss_weight = (
                torch.cat(
                    (
                        torch.ones(self.batch_size) * self.neighbor_model_alpha,
                        torch.ones(self.batch_size) * (1 - self.neighbor_model_alpha),
                    )
                )
                .view((-1, 1))
                .to(device)
            )
        self.expert_reward_ones = (
            torch.ones(self.batch_size).view((-1, 1)) * self.reward_scaling_weight
        )
        self.expert_reward_ones = self.expert_reward_ones.to(device)

    def start(self, agent, env, storage, util_dict):
        if self.oracle_neighbor:
            self.NeighborhoodNet = util_dict["OracleNeighborhoodNet"].to(device)
        else:
            self.NeighborhoodNet = util_dict["NeighborhoodNet"].to(device)
            if self.use_pretrained_neighbor:
                self.NeighborhoodNet.load_state_dict(
                    torch.load(self.pretrained_neighbor_weight_path)[
                        "neighborhood_state_dict"
                    ]
                )
                print(
                    f"load pretrained neighbor model from {self.pretrained_neighbor_weight_path}"
                )
            self.NeighborhoodNet_optimizer = torch.optim.Adam(
                self.NeighborhoodNet.parameters(), lr=3e-4
            )
        if self.use_discriminator:
            self.Discriminator = util_dict["Discriminator"].to(device)
            self.Discriminator_optimizer = torch.optim.Adam(
                self.Discriminator.parameters(), lr=3e-4
            )
            self.Discriminator_criteria = nn.BCELoss(reduction="none")
        if self.use_IDM:
            self.InverseDynamicModule = util_dict["InverseDynamicModule"].to(device)
            self.IDM_optimizer = torch.optim.Adam(
                self.InverseDynamicModule.parameters(), lr=3e-4
            )
            self.IDM_criteria = nn.MSELoss()
        if self.use_target_neighbor:
            self.TargetNeighborhoodNet = copy.deepcopy(self.NeighborhoodNet).to(device)
        self.expert_data_num = storage.load_expert_data(
            algo=self.algo,
            env_id=self.env_id,
            duplicate_expert_last_state=self.duplicate_expert_last_state,
            data_name=self.data_name,
            expert_sub_sample_ratio=self.expert_sub_sample_ratio,
            only_use_relative_state=self.only_use_relative_state,
        )
        if self.reset_as_expert_state:
            self.env_reset_options = storage.env_reset_options
        self.gen_data(storage)
        self.train(agent, env, storage)

    def test(self, agent, env, render_id=0):
        # agent.eval()
        total_reward = []
        total_neighborhood_reward = []
        total_full_reward = []
        render = self.render and render_id % 40 == 0
        if render:
            frame_buffer = []
            if not os.path.exists(f"./experiment_logs/{self.env_id}/{self.log_name}/"):
                os.makedirs(f"./experiment_logs/{self.env_id}/{self.log_name}/")
        if self.record_success_rate:
            success_counter = 0.0
        for i in range(10):
            state_dim = env.get_observation_dim()
            traj_ns = np.ones((1000, state_dim))
            traj_sa = np.ones((1000, state_dim + self.action_dim))
            mask = np.zeros(1000)
            step_counter = 0
            try:
                state = env.reset(testing=True)
            except:
                state = env.reset()
            if self.reset_as_expert_state:
                if len(self.initial_state_key_to_add_noise) > 0:
                    for key in self.initial_state_key_to_add_noise:
                        self.env_reset_options["initial_state_dict"][
                            key
                        ] += np.random.normal(
                            0,
                            self.initial_state_noise_std,
                            self.env_reset_options["initial_state_dict"][key].shape,
                        )
                state = env.reset(options=self.env_reset_options)
            else:
                state = env.reset()

            done = False
            episode_reward = 0
            if self.ood:
                for _ in range(5):
                    state, reward, done, info = env.step(
                        env.action_space.sample()
                    )  # env.action_space.sample()
            while not done:
                action = agent.act(state, testing=True)
                # agent.q_network.reset_noise()
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                if render:
                    frame_buffer.append(env.render(mode="rgb_array"))
                if not self.is_continuous:
                    #turn action into one hot
                    action = np.eye(self.action_dim)[action]
                traj_sa[step_counter] = np.hstack((state, action))
                state = next_state
                traj_ns[step_counter] = next_state
                step_counter += 1
            if self.record_success_rate:
                if state[-1] == 1:
                    success_counter += 1
            total_reward.append(episode_reward)

            mask[:step_counter] = 1
            traj_sa = torch.FloatTensor(traj_sa).to(device)
            traj_ns = torch.FloatTensor(traj_ns).to(device)
            cartesian_product_state = torch.cat(
                (
                    torch.repeat_interleave(traj_ns, self.expert_data_num, dim=0),
                    # in mujoco, 1000 is the number of expert states
                    self.testing_expert_ns_data.reshape((-1, state_dim)),
                ),
                dim=1,
            )
            with torch.no_grad():
                prob = self.NeighborhoodNet(cartesian_product_state)
                if self.use_discriminator:
                    gail_prob = self.Discriminator(traj_sa)
                    gail_reward = gail_prob.cpu().numpy().sum()
            prob = prob.reshape((1000, self.expert_data_num)).sum(dim=1)
            prob = prob.cpu().numpy() * mask
            reward = prob.sum()
            total_neighborhood_reward.append(reward)
            if self.use_discriminator:
                total_full_reward.append((1-self.beta)*reward + self.beta*gail_reward)
        with torch.no_grad():
            expert_prob = self.NeighborhoodNet(self.expert_cartesian_product_state)
        expert_reward = expert_prob.cpu().numpy().sum()
        total_neighborhood_reward = np.array(total_neighborhood_reward)
        total_neighborhood_reward_mean = total_neighborhood_reward.mean()
        total_neighborhood_reward_std = total_neighborhood_reward.std()
        total_neighborhood_reward_min = total_neighborhood_reward.min()
        total_neighborhood_reward_max = total_neighborhood_reward.max()
        if self.use_discriminator:
            total_full_reward = np.array(total_full_reward)
        if render:
            imageio.mimsave(
                f"./experiment_logs/{self.env_id}/{self.log_name}/{render_id}.gif",
                frame_buffer,
            )
        total_reward = np.array(total_reward)
        total_reward_mean = total_reward.mean()
        total_reward_std = total_reward.std()
        total_reward_min = total_reward.min()
        total_reward_max = total_reward.max()
        # agent.train()
        if self.record_success_rate:
            success_counter /= 10.0
        return_info = {
            "testing_reward_mean": total_reward_mean,
            "testing_reward_std": total_reward_std,
            "testing_reward_min": total_reward_min,
            "testing_reward_max": total_reward_max,
            "neighborhood_agent_reward_mean": total_neighborhood_reward_mean,
            "neighborhood_expert_reward": expert_reward,
            "neighborhood_agent_reward_std": total_neighborhood_reward_std,
            "neighborhood_agent_reward_min": total_neighborhood_reward_min,
            "neighborhood_agent_reward_max": total_neighborhood_reward_max,
            "relative_neighborhood_agent_reward": total_neighborhood_reward_mean
            / expert_reward,
        }
        if self.record_success_rate:
            return_info["success_rate"] = success_counter
        if self.use_discriminator:
            return_info["relative_full_reward"] = total_full_reward.mean() / ((1-self.beta)*expert_reward+(self.beta)*1000)
            return_info["full_reward_mean"] = total_full_reward.mean()
        return return_info

    def train(self, agent, env, storage):
        self.is_continuous = env.is_continuous()
        self.action_dim = env.get_action_dim()
        if self.bc_pretraining and not self.state_only:
            for _ in range(50000):
                (expert_state, expert_action, _, _, _) = storage.sample(
                    self.batch_size,
                    expert=True,
                )
                if not storage.to_tensor:
                    expert_state = torch.FloatTensor(expert_state)
                    expert_action = torch.FloatTensor(expert_action)
                expert_state = expert_state.to(device)
                expert_action = expert_action.to(device)
                bc_loss = agent.bc_update(expert_state, expert_action, use_mu=False)
            print(f"BC pretraining finished, BC loss:{bc_loss}")
        if self.buffer_warmup:
            if self.reset_as_expert_state:
                if len(self.initial_state_key_to_add_noise) > 0:
                    for key in self.initial_state_key_to_add_noise:
                        self.env_reset_options["initial_state_dict"][
                            key
                        ] += np.random.normal(
                            0,
                            self.initial_state_noise_std,
                            self.env_reset_options["initial_state_dict"][key].shape,
                        )
                state = env.reset(options=self.env_reset_options)
            else:
                state = env.reset()
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                if done:
                    if self.reset_as_expert_state:
                        if len(self.initial_state_key_to_add_noise) > 0:
                            for key in self.initial_state_key_to_add_noise:
                                self.env_reset_options["initial_state_dict"][
                                    key
                                ] += np.random.normal(
                                    0,
                                    self.initial_state_noise_std,
                                    self.env_reset_options["initial_state_dict"][
                                        key
                                    ].shape,
                                )
                        state = env.reset(options=self.env_reset_options)
                    else:
                        state = env.reset()
                    done = False
                else:
                    state = next_state
        self.best_testing_reward = -1e7
        self.best_testing_neighborhood_reward = -1e7
        best_episode = 0
        print("warmup finished")
        episode = 0
        timesteps = 0
        while True:
            if (
                not self.oracle_neighbor
                and episode % self.update_neighbor_frequency == 0
                and episode <= self.update_neighbor_until
                and not self.use_target_neighbor
                and not self.use_pretrained_neighbor
            ):
                for _ in range(self.update_neighbor_step):
                    neighbor_loss = self.update_neighbor_model(storage)
                wandb.log({"neighbor_model_loss": neighbor_loss}, commit=False)
            if self.hybrid and np.random.rand() < 0.2:
                state, _, _, _, done, expert_env_state = storage.sample(
                    batch_size=1,
                    expert=True,
                    return_expert_env_states=True,
                    exclude_tail_num=1,
                )
                env.reset()
                env.sim.set_state(expert_env_state)
                env.sim.forward()
                state = state[0]
            else:
                if self.fix_env_random_seed:
                    state = env.reset(seed=0)
                else:
                    if self.reset_as_expert_state:
                        if len(self.initial_state_key_to_add_noise) > 0:
                            for key in self.initial_state_key_to_add_noise:
                                self.env_reset_options["initial_state_dict"][
                                    key
                                ] += np.random.normal(
                                    0,
                                    self.initial_state_noise_std,
                                    self.env_reset_options["initial_state_dict"][
                                        key
                                    ].shape,
                                )
                        state = env.reset(options=self.env_reset_options)
                    else:
                        state = env.reset()
            done = False
            total_reward = 0
            while not done:
                if hasattr(env, "eval_toy_q") and timesteps % 200 == 0:
                    '''
                    only for toy env
                    '''
                    _, _, _, testing_step_penalty = env.eval_toy_q(
                        agent,
                        f"./experiment_logs/{self.env_id}{self.log_name}/",
                        timesteps,
                        storage,
                        self.NeighborhoodNet,
                        self.oracle_neighbor,
                    )
                    wandb.log(
                        {
                            "eval_step_penalty": testing_step_penalty,
                            "eval_total_steps": timesteps,
                        }
                    )

                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                state = next_state
                neighbor_info = {}
                if self.use_target_neighbor and not self.use_pretrained_neighbor:
                    neighbor_info = self.update_neighbor_model(storage)
                if self.use_IDM:
                    IDM_info = self.update_IDM(storage)
                else:
                    IDM_info = {}
                if self.use_discriminator:
                    discriminator_info=self.update_discriminator(storage)
                else:
                    discriminator_info={}
                loss_info = agent.update_using_neighborhood_reward(
                    storage,
                    self.NeighborhoodNet
                    if not self.use_target_neighbor
                    else self.TargetNeighborhoodNet,
                    self.expert_ns_data,
                    self.expert_reward_ones,
                    self.margin_value,
                    self.bc_only,
                    self.no_bc,
                    self.oracle_neighbor,
                    self.discretize_reward,
                    self.policy_threshold_ratio,
                    self.use_env_done,
                    self.use_relative_reward,
                    self.state_only,
                    self.critic_without_entropy,
                    self.target_entropy_weight,
                    self.reward_scaling_weight,
                    self.use_true_expert_relative_reward,
                    self.use_top_k,
                    self.k_of_topk,
                    self.InverseDynamicModule if self.use_IDM else None,
                    self.complementary_reward,
                    self.Discriminator if self.use_discriminator else None,
                    self.beta
                )
                timesteps += 1
                
            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": episode,
                    "buffer_size": len(storage),
                    "threshold_ratio": self.policy_threshold_ratio,
                    **loss_info,
                    **neighbor_info,
                    **IDM_info,
                    **discriminator_info,
                    "total_steps": timesteps,
                }
            )
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()

            if episode % 5 == 0 and episode > 0:
                testing_reward = self.test(
                    agent, env, render_id=episode if self.render else None
                )
                if testing_reward["testing_reward_mean"] > self.best_testing_reward:
                    self.best_testing_reward = testing_reward["testing_reward_mean"]
                    self.save_model_weight(agent, episode)

                # neighbor_testing_reward = self.test_with_neighborhood_model(agent, env)
                if (
                    testing_reward["relative_neighborhood_agent_reward"]
                    > self.best_testing_neighborhood_reward
                ):
                    self.best_testing_neighborhood_reward = testing_reward[
                        "relative_neighborhood_agent_reward"
                    ]
                    self.save_model_weight(agent, episode, oracle_reward=False)
                wandb.log(
                    {
                        **testing_reward,
                        "testing_episode_num": episode,
                        "testing_total_steps": timesteps,
                        "best_testing_reward": self.best_testing_reward,
                        "best_testing_neighborhood_reward": self.best_testing_neighborhood_reward,
                    }
                )

            if self.auto_threshold_ratio:
                self.policy_threshold_ratio *= self.threshold_discount_factor
            episode += 1
            if timesteps >= self.total_timesteps and episode >= self.episodes:
                break

    def save_model_weight(self, agent, episode, oracle_reward=True):
        if oracle_reward:
            best_reward = self.best_testing_reward
        else:
            best_reward = self.best_testing_neighborhood_reward
        agent.cache_weight()
        agent.save_weight(
            best_testing_reward=best_reward,
            algo="neighborhood_il",
            env_id=self.env_id,
            episodes=episode,
            log_name=self.log_name + ("_oracle" if oracle_reward else "_neighbor"),
            oracle_reward=oracle_reward,
        )
        path = f"./trained_model/neighborhood/{self.env_id}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "episodes": episode,
            "neighborhood_state_dict": self.NeighborhoodNet.state_dict(),
            "neighborhood_optimizer_state_dict": self.NeighborhoodNet_optimizer.state_dict(),
        }

        file_path = os.path.join(
            path,
            f"episode{episode}_reward{round(best_reward,4)}_{self.log_name}"
            + ("_oracle" if oracle_reward else "_neighbor")
            + ".pt",
        )
        torch.save(data, file_path)
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

    def update_neighbor_model(self, storage):
        state, action, reward, next_state, done = storage.sample(
            self.batch_size, only_last_1m=not self.infinite_neighbor_buffer
        )
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
        # calculate accuracy
        prediction = prediction > 0.5
        result = prediction == self.update_neighor_label
        positive_accuracy = torch.sum(result[: len(posivite_data)]) / len(posivite_data)
        if not self.hard_negative_sampling:
            easy_negative_accuracy = torch.sum(result[len(posivite_data) :]) / len(
                negative_data
            )
            with torch.no_grad():
                negative_hard_data = torch.cat((next_state, state), axis=1)
                negative_hard_data = negative_hard_data.to(device)
                prediction = self.NeighborhoodNet(negative_hard_data)
                prediction = prediction > 0.5
                result = prediction == False
                hard_negative_accuracy = torch.sum(result) / len(negative_hard_data)

        else:
            easy_negative_accuracy = torch.sum(
                result[len(posivite_data) : -len(negative_hard_data)]
            ) / len(negative_data)
            hard_negative_accuracy = torch.sum(
                result[-len(negative_hard_data) :]
            ) / len(negative_hard_data)
        return {
            "neighbor_model_loss": loss.item(),
            "neighbor_model_positive_accuracy": positive_accuracy.item(),
            "neighbor_model_easy_negative_accuracy": easy_negative_accuracy.item(),
            "neighbor_model_hard_negative_accuracy": hard_negative_accuracy.item(),
        }

    def update_target_neighbor_model(self):
        for target_param, param in zip(
            self.TargetNeighborhoodNet.parameters(), self.NeighborhoodNet.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.neighborhood_tau)
                + param.data * self.neighborhood_tau
            )

    def update_IDM(self, storage):
        # code for training inverse dynamic module
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            next_state = torch.FloatTensor(next_state)
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        input_data = torch.cat((state, next_state), axis=1)
        input_data = input_data.to(device)
        prediction = self.InverseDynamicModule(input_data)
        loss = self.IDM_criteria(prediction, action)
        loss = torch.mean(loss)
        self.IDM_optimizer.zero_grad()
        loss.backward()
        self.IDM_optimizer.step()
        return {"IDM_loss": loss.item()}

    def update_discriminator(self, storage):
        state, action, _, _, _ = storage.sample(
            self.batch_size
        )
        expert_state, expert_action, _, _, _ = storage.sample(
            self.batch_size, expert=True
        )
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            expert_state = torch.FloatTensor(expert_state)
            expert_action = torch.FloatTensor(expert_action)
        state = state.to(device)
        # if is discrete, turn action into onehot
        if not self.is_continuous:
            action = F.one_hot(action.to(torch.int64), num_classes=self.action_dim)
        action = action.to(device)
        expert_state = expert_state.to(device)
        expert_action = expert_action.to(device)
        expert_data = torch.cat((expert_state, expert_action), axis=1)
        agent_data = torch.cat((state, action), axis=1)
        expert_prediction = self.Discriminator(expert_data)
        agent_prediction = self.Discriminator(agent_data)
        expert_loss = self.Discriminator_criteria(expert_prediction, torch.ones_like(expert_prediction))
        expert_loss = torch.mean(expert_loss)
        agent_loss = self.Discriminator_criteria(agent_prediction, torch.zeros_like(agent_prediction))
        agent_loss = torch.mean(agent_loss)
        loss = expert_loss + agent_loss
        self.Discriminator_optimizer.zero_grad()
        loss.backward()
        self.Discriminator_optimizer.step()
        return {"discriminator_loss": loss.item()}
        
