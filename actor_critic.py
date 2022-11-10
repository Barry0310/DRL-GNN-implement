import numpy as np
import torch
from Actor import Actor
from Critic import Critic
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions


class PPOAC:
    def __init__(self, hyper_parameter):
        H = hyper_parameter
        self.gae_gamma = H['gae_gamma']
        self.gae_lambda = H['gae_lambda']
        self.clip_value = H['clip_value']
        self.mini_batch = H['mini_batch']
        self.feature_size = H['feature_size']
        self.entropy_beta = H['entropy_beta']
        self.actor = Actor(feature_size=self.feature_size, t=H['t'], readout_units=H['readout_units'])
        self.critic = Critic(feature_size=self.feature_size, t=H['t'], readout_units=H['readout_units'])
        self.optimizer = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=H['lr'], eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=H['lr_decay_step'], gamma=H['lr_decay_rate'])

        self.buffer = []

    def old_cummax(self, alist, extractor):
        maxes = [np.amax(extractor(v)) + 1 for v in alist]
        cummaxes = [np.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(np.array(np.sum(maxes[0:i + 1])))
        return cummaxes

    def predict(self, env, src, dst):
        list_k_features = []

        middle_point_list = env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]
        for mid in range(len(middle_point_list)):
            env.mark_action_sp(src, middle_point_list[mid], src, dst)
            if middle_point_list[mid] != dst:
                env.mark_action_sp(middle_point_list[mid], dst, src, dst)
            features = self.actor_get_graph_features(env)
            list_k_features.append(features)
            env.edge_state[:, 2] = 0

        graph_ids = [np.full([list_k_features[it]['link_state'].shape[0]], it) for it in range(len(list_k_features))]

        first_offset = self.old_cummax(list_k_features, lambda v: v['first'])
        second_offset = self.old_cummax(list_k_features, lambda v: v['second'])

        tensor = ({
            'graph_id': np.concatenate([v for v in graph_ids], axis=0, dtype='int64'),
            'link_state': np.concatenate([v['link_state'] for v in list_k_features], axis=0, dtype='float32'),
            'first': np.concatenate([v['first'] + m for v, m in zip(list_k_features, first_offset)], axis=0,
                                    dtype='int64'),
            'second': np.concatenate([v['second'] + m for v, m in zip(list_k_features, second_offset)], axis=0,
                                     dtype='int64'),
            'state_dim': self.feature_size,
            'num_actions': len(middle_point_list),
        })
        q_values = self.actor(tensor)
        q_values = torch.reshape(q_values, (-1, ))
        soft_max_q_values = torch.nn.Softmax(q_values)

        return soft_max_q_values, tensor

    def actor_get_graph_features(self, env):
        temp = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': env.edge_state[:,2],
            'utilization': np.divide(env.edge_state[:,0], env.edge_state[:, 1]),
            'first': env.first,
            'second': env.second
        }

        temp['utilization'] = np.reshape(temp['utilization'][0:temp['num_edges']], [temp['num_edges'], 1])
        temp['capacity'] = np.reshape(temp['capacity'][0:temp['num_edges']], [temp['num_edges'], 1])
        temp['bw_allocated'] = np.reshape(temp['bw_allocated'][0:temp['num_edges']], [temp['num_edges'], 1])

        hidden_states = np.concatenate([temp['utilization'], temp['capacity'], temp['bw_allocated']], axis=1)
        link_state = np.pad(hidden_states, ((0, 0), (0, self.feature_size - 3)), 'constant', constant_values=(0, ))

        inputs = {'link_state': link_state, 'first': temp['first'][0:temp['length']],
                  'second': temp['second'][0:temp['length']]}

        return inputs

    def critic_get_graph_features(self, env):
        temp = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'utilization': np.divide(env.edge_state[:, 0], env.edge_state[:, 1]),
            'first': env.first,
            'second': env.second
        }

        temp['utilization'] = np.reshape(temp['utilization'][0:temp['num_edges']], [temp['num_edges'], 1])
        temp['capacity'] = np.reshape(temp['capacity'][0:temp['num_edges']], [temp['num_edges'], 1])

        hidden_states = np.concatenate([temp['utilization'], temp['capacity']], axis=1)
        link_state = np.pad(hidden_states, ((0, 0), (0, self.feature_size - 2)), 'constant', constant_values=(0,))

        inputs = {'link_state': np.array(link_state, dtype='float32'),
                  'first': np.array(temp['first'][0:temp['length']], dtype='int64'),
                  'second': np.array(temp['second'][0:temp['length']], dtype='int64'),
                  'state_dim': self.feature_size}

        return inputs

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
        loss = - (advantages.detach() * torch.log(action_probs)).sum()
        print('actor loss:', loss)
        return loss

    def compute_critic_loss(self, returns, c_vals):
        loss = F.mse_loss(returns.detach(), c_vals).sum()
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
