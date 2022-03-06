import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, feature_size=20, t=4, readout_units=20):
        super(MPNN, self).__init__()
        self.feature_size = feature_size
        self.t = t
        self.readout_units = readout_units
        self.message = nn.Sequential(
            nn.Linear(feature_size*2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
        )
        self.update = nn.GRUCell(input_size=feature_size, hidden_size=feature_size)
        self.readout = nn.Sequential(
            nn.Linear(feature_size, self.readout_units),
            nn.ReLU(),
            nn.Linear(self.readout_units, self.readout_units),
            nn.ReLU(),
            nn.Linear(self.readout_units, 1)
        )

    def forward(self, x):
        state = torch.tensor(x['link_state'])
        pair = torch.tensor(x['pair'])
        index = torch.stack([i for i in pair[::2]])

        for _ in range(self.t):
            tmp = torch.gather(state, 0, pair)
            neighbor = torch.stack([torch.cat((i, j), 0) for i, j in zip(tmp[::2], tmp[1::2])])

            m = self.message(neighbor.float())

            m = torch.zeros(state.shape, dtype=m.dtype).scatter_add_(0, index, m)

            state = self.update(m, state.float())

        feature = torch.sum(state, 0)
        output = self.readout(feature)

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