import wandb
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RunningStat(object):
    """
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class rewardProcessor:
    """
    codes from https://github.com/MadryLab/implementation-matters/blob/5ee6ecb12545365d9178135e65576adfc0d82f52/src/policy_gradients/torch_utils.py#L408
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x):
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff / (self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x


# some code comes from https://www.youtube.com/watch?v=HR8kQMTO8bk
class vanilla_on_policy_training_stage:
    def __init__(self, config):
        self.episodes = config.episodes
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        self.max_episode_step = config.max_episode_step
        self.train_agnet_period = config.train_agnet_period
        self.standarize_reward = config.standarize_reward
        wandb.init(
            project="RL_Implementation",
            name=f"{self.algo}_{self.env_id}{config.log_name}",
            config=config,
        )

    def test(self, agent, env):
        agent.eval()
        total_reward = 0
        for i in range(3):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
        total_reward /= 3
        agent.train()
        return total_reward

    def start(self, agent, env, storage):
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        if self.continue_training:
            agent.load_weight(self.env_id)

        best_testing_reward = -1e7
        best_episode = 0
        total_steps = 0
        for i in range(self.episodes):
            state = env.reset()
            value = agent.get_state_value(state)
            done = False
            total_reward = 0
            if self.standarize_reward:
                rp = rewardProcessor(())
            for _ in range(self.max_episode_step):
                action, log_prob = agent.act(state)
                next_state, reward, done, info = env.step(action)
                reward = rp(reward) if self.standarize_reward else reward
                total_steps += 1
                total_reward += reward
                next_value = agent.get_state_value(next_state)
                storage.store(state, action, reward, value, next_value, log_prob, done)

                state = next_state
                value = next_value
                if total_steps % self.train_agnet_period == 0:
                    storage.calculate_gae(gamma=agent.gamma, decay=agent.lambda_decay)
                    storage.discount_rewards(gamma=agent.gamma, critic=agent.critic)
                    loss_info = agent.update(storage)
                    wandb.log(
                        {
                            **loss_info,
                            "training_reward": total_reward,
                            "episode_num": i,
                            "total_steps": total_steps,
                        }
                    )
                    storage.clear()
                if total_steps % agent.action_std_decay_freq == 0:
                    agent.update_std()
                if done:
                    break

            if i % 5 == 0:
                testing_reward = self.test(agent, env)
                if testing_reward > best_testing_reward:
                    agent.cache_weight()
                    best_testing_reward = testing_reward
                    best_episode = i
                wandb.log({"testing_reward": testing_reward, "testing_episode_num": i})
            if i % self.save_weight_period == 0:
                agent.save_weight(
                    best_testing_reward, self.algo, self.env_id, best_episode
                )
        agent.save_weight(best_testing_reward, self.algo, self.env_id, best_episode)
