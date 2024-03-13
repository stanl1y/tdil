import wandb
import numpy as np
import os
import imageio


class evaluate:
    def __init__(self, config):
        self.episodes = config.episodes
        self.algo = config.algo
        self.env_id = config.env
        self.render = config.render
        self.ood = config.ood
        self.perturb_from_mid = config.perturb_from_mid
        self.weight_path = config.weight_path
        self.perturb_step_num = config.perturb_step_num
        self.perturb_with_repeated_action = config.perturb_with_repeated_action
        self.log_name = config.log_name
        log_name = f"{self.algo}_{self.env_id}_eval"
        if self.ood:
            log_name += "_ood"
            if self.perturb_from_mid:
                log_name += "_perturb_from_mid"
            log_name += f"{self.perturb_step_num}"
            if self.perturb_with_repeated_action:
                log_name += "_perturb_with_repeated_action"
        log_name += f"_weight{self.weight_path.split('_')[-1]}"
        log_name += self.log_name
        wandb.init(
            project="Evaluate_IL",
            name=log_name,
            config=config,
        )

    def test(self, agent, env):
        agent.eval()
        render = self.render
        save_dir = ""
        if render:
            frame_buffer = []
            if self.ood:
                if not os.path.exists(
                    f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
                ):
                    os.makedirs(
                        f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
                    )
                save_dir = f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
            else:
                if not os.path.exists(
                    f"./experiment_logs/{self.env_id}/{self.algo}_eval/"
                ):
                    os.makedirs(f"./experiment_logs/{self.env_id}/{self.algo}_eval/")
                save_dir = f"./experiment_logs/{self.env_id}/{self.algo}_eval/"
        reward_list = []
        for i in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            if self.ood:
                if self.perturb_from_mid:
                    for _ in range(500):
                        action = agent.act(state, testing=True)
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        if render:
                            frame_buffer.append(env.render(mode="rgb_array"))
                else:
                    action = agent.act(state, testing=True)
                for _ in range(self.perturb_step_num):
                    if self.perturb_with_repeated_action:
                        state, reward, done, info = env.step(action)
                    else:
                        state, reward, done, info = env.step(env.action_space.sample())
                    total_reward += reward
                    if render:
                        frame_buffer.append(env.render(mode="rgb_array"))
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                if render:
                    frame_buffer.append(env.render(mode="rgb_array"))
                total_reward += reward
                state = next_state
            reward_list.append(total_reward)
            # if self.ood:
            #     for _ in range(10):
            #         state, reward, done, info = env.step(
            #             env.action_space.sample()
            #         )  # env.action_space.sample()
            #         if render:
            #             frame_buffer.append(env.render(mode="rgb_array"))
            #         total_reward += reward
            # while not done:
            #     action = agent.act(state, testing=True)
            #     next_state, reward, done, info = env.step(action)
            #     if render:
            #         frame_buffer.append(env.render(mode="rgb_array"))
            #     total_reward += reward
            #     state = next_state
            if render:
                imageio.mimsave(
                    f"{save_dir}{i}.gif",
                    frame_buffer,
                )
                frame_buffer = []
            wandb.log(
                {
                    "testing_reward": total_reward,
                    "episode_num": i,
                }
            )
        reward_list.remove(max(reward_list))
        reward_list.remove(min(reward_list))
        wandb.log(
            {
                "testing_reward_mean": np.mean(reward_list),
                "testing_reward_std": np.std(reward_list),
                "testing_reward_max": np.max(reward_list),
                "testing_reward_min": np.min(reward_list),
            }
        )


    def start(self, agent, env, storage):
        if self.weight_path:
            agent.load_weight(path=self.weight_path)
        else:
            agent.load_weight(algo=self.algo, env_id=self.env_id)
        agent.eval()
        self.test(agent, env)
