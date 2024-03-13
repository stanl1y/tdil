from .base_dqn import base_dqn
import torch
import os
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class rainbow_dqn(base_dqn):
    def __init__(
        self,
        observation_dim,
        action_num,
        hidden_dim=256,
        gamma=0.99,
        optim="adam",
        network_type="vanilla",
        noisy_network=False,
        double_dqn=False,
        soft_update_target=True,
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        lr=3e-4,
        tau=0.01,
        batch_size=256,
    ):

        super().__init__(
            observation_dim=observation_dim,
            action_num=action_num,
            hidden_dim=hidden_dim,
            network_type=network_type,
            noisy_network=noisy_network,
            soft_update_target=soft_update_target,
            gamma=gamma,
            optim=optim,
            lr=lr,
            tau=tau,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
        )
        self.double_dqn = double_dqn

    def act(self, state, testing=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.rand() < self.epsilon and not testing and not self.noisy_network:
            action = np.random.randint(0, self.action_num)
        else:#acting greedly
            action = self.q_network(state).cpu().detach().numpy()
            action = np.argmax(action[0])
        return action

    def update_dqn(self, state, action, reward, next_state, done):
        """update noisy network noise"""
        if self.noisy_network:
            self.q_network.reset_noise()
            self.q_network_target.reset_noise()

        """compute target value"""
        with torch.no_grad():
            target_q_val = self.q_network_target(next_state)
            if not self.double_dqn:
                target_q_val = target_q_val.max(1)[0].unsqueeze(1)
            else:
                next_action = self.q_network(next_state).argmax(1).unsqueeze(1)
                target_q_val = target_q_val.gather(1, next_action)
            target_q_val = reward + (1 - done) * self.gamma * target_q_val

        """compute loss and update"""
        pred = self.q_network(state)
        pred = pred.gather(1, action)
        loss = self.criterion(pred, target_q_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def DQfD_update(self, expert_state, expert_action, margin_value=0.1):
        q_val = self.q_network(expert_state)
        max_action = self.q_network(expert_state).argmax(1).unsqueeze(1)
        max_action_q_val = q_val.gather(1, max_action)
        expert_action_q_val = q_val.gather(1, expert_action)
        loss = (
            max_action_q_val + (max_action != expert_action) * margin_value
        ) - expert_action_q_val
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def neighborhood_reward(self, NeighborhoodNet, storage, next_state):
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
                np.repeat(next_state, len(expert_state), axis=0),
                np.tile(expert_state, (len(next_state), 1)),
            ),
            axis=1,
        )
        cartesian_product_state = torch.FloatTensor(cartesian_product_state).to(device)

        with torch.no_grad():
            prob = NeighborhoodNet(cartesian_product_state)
            # prob is in the shape of (len(next_state)*len(expert_state), 1)
        reward = prob.reshape((len(next_state), len(expert_state))).sum(
            axis=1, keepdims=True
        )
        return reward

    def update_using_neighborhood_reward(
        self, storage, NeighborhoodNet, margin_value=0.1
    ):
        """sample agent data"""
        state, action, _, next_state, done = storage.sample(self.batch_size)

        reward = self.neighborhood_reward(NeighborhoodNet, storage, next_state)

        state = torch.FloatTensor(state).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.zeros_like(done)).to(
            device
        )  # try to use no done in imitation learning


        td_loss = self.update_dqn(state, action, reward, next_state, done)

        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            self.batch_size, expert=True
        )
        expert_state = torch.FloatTensor(expert_state).to(device)
        expert_action = torch.tensor(expert_action, dtype=torch.int64).to(device)
        dqfd_loss = self.DQfD_update(
            expert_state, expert_action, margin_value=margin_value
        )
        self.update_target()
        return {"td_loss": td_loss, "dqfd_loss": dqfd_loss}

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
        loss = self.update_dqn(state, action, reward, next_state, done)
        self.update_target()
        return {
            "loss": loss,
        }  # this dict will be recorded by wandb

    def cache_weight(self):
        self.best_q_network.load_state_dict(self.q_network.state_dict())
        self.best_optimizer.load_state_dict(self.optimizer.state_dict())

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
            "dqn_state_dict": self.best_q_network.cpu().state_dict(),
            "optimizer_state_dict": self.best_optimizer.state_dict(),
            "reward": best_testing_reward,
        }

        torch.save(data, file_path)
        try:
            os.remove(self.previous_checkpoint_path)
        except:
            pass
        self.previous_checkpoint_path = file_path

    def load_weight(self, algo="rainbow_dqn", env_id=None, path=None):
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

        self.q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.q_network = self.q_network.to(device)
        self.q_network_target.load_state_dict(checkpoint[f"dqn_state_dict"])
        self.q_network_target = self.q_network_target.to(device)
        self.best_q_network.load_state_dict(checkpoint["dqn_state_dict"])
        self.best_q_network = self.best_q_network.to(device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("loaded weight from", path)
