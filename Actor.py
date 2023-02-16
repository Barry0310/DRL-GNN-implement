import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, feature_size=20, t=4, readout_units=20):
        super(Actor, self).__init__()
        self.feature_size = feature_size
        self.t = t
        self.readout_units = readout_units
        self.message = nn.GRU(input_size=feature_size, hidden_size=feature_size, batch_first=True)
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
        torch.nn.init.orthogonal_(self.out_layer.weight, gain=np.sqrt(0.01))
        torch.nn.init.constant_(self.out_layer.bias, 0)

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
        link_state = x['link_state']
        path_state = x['path_state']

        link_id = x['link_id'].unsqueeze(1).expand(-1, self.feature_size)
        path_seq = torch.stack([x['path_id'], x['sequence']], dim=1)
        max_len = max(x['sequence']) + 1

        for _ in range(self.t):
            link_to_path = torch.gather(link_state, 0, link_id)
            message_input = torch.zeros((x['num_actions'], max_len, self.feature_size),
                                        device=link_to_path.device, dtype=torch.float32)
            message_input[path_seq[:, 0], path_seq[:, 1]] = link_to_path

            m, new_p_s = self.message(message_input, path_state.unsqueeze(0))

            path_state = new_p_s.squeeze(0)

            m = torch.zeros(link_state.shape, device=link_state.device).scatter_add_(0, link_id, m[list(path_seq.T)])

            link_state = self.update(m, link_state)

        output = self.out_layer(self.readout(path_state))

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