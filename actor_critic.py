import torch
import torch.nn as nn
from gnn_model import MPNN
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, feature_size, t, readout_units, lr):
        super(Actor, self).__init__()
        self.model = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        action_distrib = []
        for action in x:
            action_distrib.append(self.model(action))

        return action_distrib


class Critic(nn.Module):
    def __init__(self, feature_size, t, readout_units, lr):
        super(Critic, self).__init__()
        self.model = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        critic_value = []
        for action in x:
            critic_value.append(self.model(action))

        return critic_value


class AC:
    def __init__(self, hyper_parameter):
        H = hyper_parameter
        self.actor = Actor(feature_size=H['feature_size'], t=H['t'], readout_units=H['readout_units'], lr=H['lr'])
        self.critic = Critic(feature_size=H['feature_size'], t=H['t'], readout_units=H['readout_units'], lr=H['lr'])
        self.episode = H['episode']
        self.gae_gamma = H['gae_gamma']
        self.gae_lambda = H['gae_lambda']
        self.clip_value = H['clip_value']
        self.buffer = []

    def pred_actor_distrib(self, action_list):
        actor_distrib = self.actor(action_list)
        return torch.tensor(actor_distrib)

    def pred_critic_value(self, action_list):
        critic_value = self.critic(action_list)
        return torch.tensor(critic_value)

    def choose_action(self, action_distrib):
        action_prob = F.softmax(action_distrib)
        return torch.argmax(action_prob).item()

    def store_result(self, actor_distrib, critic_value, action, demand, done, reward):
        self.buffer.append((actor_distrib, critic_value, action, demand, done, reward))

    def compute_gae(self):
        pass

    def compute_actor_loss(self):
        pass

    def compute_critic_loss(self):
        pass

    def gradients(self):
        pass
