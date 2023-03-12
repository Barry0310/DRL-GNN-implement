from base import GNN
import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, feature_size=20, t=4, readout_units=20):
        super(Critic, self).__init__()
        self.Q1_net = GNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.Q2_net = GNN(feature_size=feature_size, t=t, readout_units=readout_units)

    def forward(self, x):
        q1 = torch.reshape(self.Q1_net(x), (-1,))
        q2 = torch.reshape(self.Q2_net(x), (-1,))

        return q1, q2

