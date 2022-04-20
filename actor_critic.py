import numpy as np
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=H['lr_decay_step'], gamma=H['lr_decay_rate'])
        self.episode = H['episode']
        self.gae_gamma = H['gae_gamma']
        self.gae_lambda = H['gae_lambda']
        self.clip_value = H['clip_value']
        self.mini_batch = H['mini_batch']
        self.feature_size = H['feature_size']
        self.entropy_beta = H['entropy_beta']
        self.buffer = []

    def _expand_dim(self, actor_input, critic_input):
        temp = [0] * (self.feature_size - 3)
        for i in actor_input:
            x = np.zeros((len(i['link_state']), self.feature_size))
            for j in range(len(i['link_state'])):
                x[j] = np.concatenate((i['link_state'][j], np.array(temp)))
            i['link_state'] = x
        y = np.zeros((len(critic_input['link_state']), self.feature_size))
        for i in range(len(critic_input['link_state'])):
            y[i] = np.concatenate((critic_input['link_state'][i], np.array(temp)))
        critic_input['link_state'] = y
        return actor_input, critic_input

    def predict(self, actor_input, critic_input):
        actor_input, critic_input = self._expand_dim(actor_input, critic_input)
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

    def buffer_size(self):
        return len(self.buffer)

    def _data_to_list(self):
        rewards = []
        c_vals = []
        done = []
        entropy = []
        action_probs = []
        for i in self.buffer:
            if i[-1] is not None:
                rewards.append(i[-1].astype('float32'))
            c_vals.append(i[1])
            if i[-2] is not None:
                done.append(not i[-2])
            if i[0] is not None:
                action_probs.append(i[0][i[2]:i[2]+1])
                entropy.append(-(torch.log(i[0])*i[0]).sum() * self.entropy_beta)
        c_vals = torch.cat(c_vals)
        action_probs = torch.cat(action_probs)
        entropy = torch.Tensor(entropy).sum()

        return rewards, c_vals, done, action_probs, entropy

    def compute_gae(self):
        rewards, c_vals, done, action_probs, entropy = self._data_to_list()

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * self.gae_gamma
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = []
        A = 0
        next_value = 0
        for r, v in zip(reversed(rewards), reversed(c_vals[1:])):
            td_error = r + next_value * self.gae_gamma - v
            A = td_error + A * self.gae_gamma * self.gae_lambda
            next_value = v
            advantages.insert(0, A)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        self.buffer_clear()
        advantages = (advantages - advantages.mean()) / advantages.std()
        returns = (returns - returns.mean()) / returns.std()

        return advantages, returns, action_probs, c_vals[1:], entropy

    def compute_actor_loss(self, advantages, action_probs):
        loss = - (advantages * torch.log(action_probs)).sum()
        print('actor loss:', loss)
        return loss

    def compute_critic_loss(self, returns, c_vals):
        loss = F.smooth_l1_loss(returns, c_vals).sum()
        print('critic loss:', loss)
        return loss

    def compute_gradients(self, loss):
        print('update weight')
        print('loss:', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
        self.optimizer.step()
        self.scheduler.step()
