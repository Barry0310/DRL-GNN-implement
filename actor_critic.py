import torch
import torch.nn as nn
from gnn_model import MPNN
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Policy(nn.Module):
    def __init__(self, feature_size, t, readout_units):
        super(Policy, self).__init__()
        self.actor = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.critic = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)

    def forward(self, x):
        action_distrib = []
        for action in x[0]:
            action_distrib.append(self.actor(action))
        critic_value = self.critic(x[1])

        return action_distrib, critic_value


class AC:
    def __init__(self, hyper_parameter):
        H = hyper_parameter
        self.model = Policy(feature_size=H['feature_size'], t=H['t'], readout_units=H['readout_units'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=H['lr'])
        self.episode = H['episode']
        self.gae_gamma = H['gae_gamma']
        self.gae_lambda = H['gae_lambda']
        self.clip_value = H['clip_value']
        self.buffer = []

    def predict(self, actor_input, critic_input):
        actor_distrib, critic_value = self.model((actor_input, critic_input))
        return torch.tensor(actor_distrib), critic_value

    def choose_action(self, action_distrib):
        """
        according to the probability of each action choose action
        """
        action_prob = F.softmax(action_distrib, dim=0)
        pa = np.array(action_prob)
        return np.random.choice(np.arange(len(pa)), size=1, p=pa)[0]

    def store_result(self, actor_distrib=None, critic_value=None, action=None, demand=None, done=None, reward=None):
        """
        replay buffer
        """
        self.buffer.append((actor_distrib, critic_value, action, demand, done, reward))

    def _data_to_list(self):
        rewards = []
        c_vals = []
        done = []
        for i in self.buffer:
            rewards.append(i[-1])
            c_vals.append(i[1].item())
            done.append(not i[-2])
        rewards.pop(-1)
        done.pop(-1)
        return rewards, c_vals, done

    def compute_gae(self):
        rewards, c_vals, done = self._data_to_list()

        size = len(rewards)
        advantages = np.zeros(size + 1)

        for i in reversed(range(size)):
            delta = rewards[i] + (self.gae_gamma * c_vals[i+1] * done[i]) - c_vals[i]
            advantages[i] = delta + (self.gae_gamma * self.gae_lambda * advantages[i+1] * done[i])

        returns = advantages[:size] + c_vals[:-1]

        return advantages[:size], returns

    def compute_actor_loss(self):
        pass

    def compute_critic_loss(self):
        pass

    def gradients(self):
        pass
