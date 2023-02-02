import torch
import torch.nn as nn
import numpy as np


class Critic(nn.Module):
    def __init__(self, feature_size=20, t=4, readout_units=20):
        super(Critic, self).__init__()
        self.feature_size = feature_size
        self.t = t
        self.readout_units = readout_units
        self.message = nn.Sequential(
            nn.Linear(feature_size*2, feature_size),
            nn.SELU()
        )
        self.message.apply(self._init_hidden_weights)
        self.update = nn.GRUCell(input_size=feature_size, hidden_size=feature_size)
        self.update.apply(self._init_hidden_weights)
        self.readout = nn.Sequential(
            nn.Linear(feature_size, self.readout_units),
            nn.SELU(),
            nn.Linear(self.readout_units, self.readout_units),
            nn.SELU()
        )
        self.readout.apply(self._init_hidden_weights)
        self.out_layer = nn.Linear(self.readout_units, 1)
        torch.nn.init.orthogonal_(self.out_layer.weight, gain=np.sqrt(1))
        torch.nn.init.constant_(self.out_layer.weight, 0)

    def _init_hidden_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GRUCell, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)

    def forward(self, x):
        state = x['link_state']
        first = x['first'].unsqueeze(1).expand(-1, x['state_dim'])
        second = x['second'].unsqueeze(1).expand(-1, x['state_dim'])

        for _ in range(self.t):
            main_edges = torch.gather(state, 0, first)
            neigh_edges = torch.gather(state, 0, second)
            edges_concat = torch.cat((main_edges, neigh_edges), 1)
            m = self.message(edges_concat)

            m = torch.zeros(state.shape, dtype=m.dtype, device=state.device).scatter_add_(0, second, m)
            state = self.update(m, state)

        feature = torch.sum(state, 0)
        output = self.out_layer(self.readout(feature))

        return output


"""

link_state:
    link_capacity: float
    link_utilization: float
    action: mark bw
    bw: float

input:
    link_state
    pair: [0, 1] => [[0, 0], [1, 1]]

"""