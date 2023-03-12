from base import GNN
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, feature_size=20, t=4, readout_units=20):
        super(Actor, self).__init__()
        self.policy_net = GNN(feature_size=feature_size, t=t, readout_units=readout_units)

    def forward(self, x):
        values = torch.reshape(self.policy_net(x), (-1, ))
        probs = nn.functional.softmax(values, dim=0)

        return probs

