from distutils.log import error


class collect_expert:
    def __init__(self, config):
        if config.expert_transition_num is not None:
            self.expert_transition_num = config.expert_transition_num
            self.based_on_transition_num = True
        elif config.expert_episode_num is not None:
            self.expert_episode_num = config.expert_episode_num
            self.based_on_transition_num = False
        else:
            raise ValueError(
                "either expert_transition_num or expert_episode_num must be set"
            )
        self.algo = config.algo
        self.env_id = config.env
        self.data_name = config.data_name
        self.weight_path = config.weight_path
        self.save_env_states = config.save_env_states

    def start(self, agent, env, storage):
        if self.weight_path:
            agent.load_weight(path=self.weight_path)
        else:
            agent.load_weight(algo=self.algo, env_id=self.env_id)
        agent.eval()
        if self.based_on_transition_num:
            done = False
            total_reward = 0
            episode_reward = 0
            episode_num = 0
            state = env.reset(seed=episode_num)
            while len(storage) < self.expert_transition_num:
                if done:
                    done = False
                    episode_num += 1
                    total_reward += episode_reward
                    episode_reward = 0
                    state = env.reset(seed=episode_num)
                env_state = env.sim.get_state() if self.save_env_states else None
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                storage.store(
                    state, action, reward, next_state, done, env_state=env_state
                )
                state = next_state
            avg_reward = round(total_reward / max(episode_num, 1), 4)
            print(f"expert data recording finish, average reward : {avg_reward}")
            storage.write_storage(
                self.based_on_transition_num,
                self.expert_transition_num,
                self.algo,
                self.env_id,
                self.data_name + str(round(avg_reward)),
            )
        else:
            total_reward = 0
            for e in range(self.expert_episode_num):
                done = False
                state = env.reset(seed=e)
                while not done:
                    env_state = env.sim.get_state() if self.save_env_states else None
                    action = agent.act(state, testing=True)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    storage.store(
                        state, action, reward, next_state, done, env_state=env_state
                    )
                    state = next_state
            avg_reward = round(total_reward / self.expert_episode_num, 4)
            print(f"expert data recording finish, average reward : {avg_reward}")
            storage.write_storage(
                self.based_on_transition_num,
                self.expert_episode_num,
                self.algo,
                self.env_id,
                self.data_name + str(round(avg_reward)),
            )
