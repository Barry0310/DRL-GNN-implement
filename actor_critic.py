import torch
import torch.nn as nn
from gnn_model import MPNN
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions


class Policy(nn.Module):
    def __init__(self, feature_size, t, readout_units):
        super(Policy, self).__init__()
        self.actor = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)
        self.critic = MPNN(feature_size=feature_size, t=t, readout_units=readout_units)

    def forward(self, x):
        action_distribs = None
        if len(x[0]):
            action_distribs = self.actor(x[0][0])
            for action in x[0][1:]:
                action_distribs = torch.cat((action_distribs, self.actor(action)), dim=0)

        critic_value = self.critic(x[1])
        return action_distribs, critic_value


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
        actor_distribs, critic_values = self.model((actor_input, critic_input))
        return actor_distribs, critic_values

    def choose_action(self, action_distrib):
        """
        according to the probability of each action choose action
        """
        action_prob = F.softmax(action_distrib, dim=0)
        pa = distributions.Categorical(action_prob)
        return pa.sample().item(), action_prob

    def store_result(self, action_prob=None, critic_value=None, action=None, demand=None, done=None, reward=None):
        """
        replay buffer
        """
        self.buffer.append((action_prob, critic_value, action, demand, done, reward))

    def buffer_clear(self):
        self.buffer = []

    def _data_to_list(self):
        rewards = []
        c_vals = []
        done = []
        action_probs = []
        for i in self.buffer:
            if i[-1] is not None:
                rewards.append(i[-1])
            c_vals.append(i[1])
            if i[-2] is not None:
                done.append(not i[-2])
            if i[0] is not None:
                action_probs.append(i[0][i[2]:i[2]+1])
        c_vals = torch.cat(c_vals)
        action_probs = torch.cat(action_probs)
        return rewards, c_vals, done, action_probs

    def compute_gae(self):
        rewards, c_vals, done, action_probs = self._data_to_list()
        size = len(rewards)
        advantages = [0] * (size + 1)

        for i in reversed(range(size)):
            delta = rewards[i] + (self.gae_gamma * c_vals[i+1] * done[i]) - c_vals[i]
            advantages[i] = delta + (self.gae_gamma * self.gae_lambda * advantages[i+1] * done[i])
        advantages = torch.tensor(advantages[:size])
        returns = advantages + c_vals[:-1]

        return advantages, returns, action_probs, c_vals[1:]

    def compute_actor_loss(self, advantages, action_probs):
        loss = - (advantages * action_probs).sum()
        return loss

    def compute_critic_loss(self, returns, c_vals):
        loss = F.smooth_l1_loss(returns, c_vals).sum()
        return loss

    def compute_gradients(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
        self.optimizer.step()
        print('sucess')
