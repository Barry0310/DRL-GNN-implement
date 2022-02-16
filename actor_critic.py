import torch
import torch.nn as nn
from gnn_model import MPNN
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, feature_size, t, readout_units):
        super(Policy, self).__init__()
        self.actor = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.critic = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)

    def forward(self, x):
        action_distrib = []
        critic_value = []
        for action in x:
            action_distrib.append(self.actor(action))
            critic_value.append(self.critic(action))

        action_prob = F.softmax(torch.stack(action_distrib))

        return action_prob, critic_value


class AC:
    def __init__(self, hyper_parameter):
        self.H = hyper_parameter
        self.model = Policy(self.H['feature_size'], self.H['t'], self.H['readout_units'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.H['lr'])
        self.gae_gamma = self.H['gae_gamma']
        self.gae_lambda = self.H['gae_lambda']
        self.clip_value = self.H['clip_value']
        self.buffer = []

    def step(self):
        pass

    def choose_action(self):
        pass

    def store_results(self):
        pass

    def apply_policy(self):
        pass

    def compute_gae(self):
        pass

    def compute_actor_loss(self):
        pass

    def compute_critic_loss(self):
        pass

    def compute_gradients(self):
        pass
