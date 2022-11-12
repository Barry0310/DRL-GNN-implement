import numpy as np
import torch
from Actor import Actor
from Critic import Critic
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import gc


class PPOAC:
    def __init__(self, hyper_parameter):
        H = hyper_parameter
        self.gae_gamma = H['gae_gamma']
        self.gae_lambda = H['gae_lambda']
        self.clip_value = H['clip_value']
        self.mini_batch = H['mini_batch']
        self.feature_size = H['feature_size']
        self.entropy_beta = H['entropy_beta']
        self.buffer_size = H['buffer_size']
        self.update_times = H['update_times']
        self.actor = Actor(feature_size=self.feature_size, t=H['t'], readout_units=H['readout_units'])
        self.critic = Critic(feature_size=self.feature_size, t=H['t'], readout_units=H['readout_units'])
        self.optimizer = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=H['lr'], eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=H['lr_decay_step'], gamma=H['lr_decay_rate'])

        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_index = np.arange(self.buffer_size)

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

        tensor = {
            'graph_id': np.concatenate([v for v in graph_ids], axis=0, dtype='int64'),
            'link_state': np.concatenate([v['link_state'] for v in list_k_features], axis=0, dtype='float32'),
            'first': np.concatenate([v['first'] + m for v, m in zip(list_k_features, first_offset)], axis=0,
                                    dtype='int64'),
            'second': np.concatenate([v['second'] + m for v, m in zip(list_k_features, second_offset)], axis=0,
                                     dtype='int64'),
            'state_dim': self.feature_size,
            'num_actions': len(middle_point_list),
        }
        q_values = self.actor(tensor)
        q_values = torch.reshape(q_values, (-1, ))
        soft_max_q_values = torch.nn.functional.softmax(q_values, dim=0)

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
        gc.collect()

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
        gc.collect()

        return inputs

    def compute_gae(self, values, masks, rewards):
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gae_gamma * values[i+1] * masks[i] - values[i]
            gae = delta + self.gae_gamma * self.gae_lambda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]

        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def _compute_actor_loss(self, adv, old_act, old_policy_probs, link_state, graph_id,
                            first, second, state_dim, num_actions):
        old_policy_probs = old_policy_probs.detach()

        q_values = self.actor({
            'graph_id': graph_id,
            'link_state': link_state,
            'first': first,
            'second': second,
            'state_dim': state_dim,
            'num_actions': num_actions,
        })
        q_values = torch.reshape(q_values, (-1,))
        new_policy_probs = torch.nn.functional.softmax(q_values, dim=0)

        ratio = torch.exp(
            torch.log(torch.sum(old_act * new_policy_probs)) - torch.log(torch.sum(old_act * old_policy_probs))
        )
        surr1 = -ratio*adv
        surr2 = -torch.clip(ratio, min=1-self.clip_value, max=1+self.clip_value) * adv

        loss = torch.max(surr1, surr2)
        entropy = -torch.sum(torch.log(new_policy_probs) * new_policy_probs)

        return loss, entropy

    def _compute_critic_loss(self, ret, link_state, first, second, state_dim):

        value = self.critic({
            'link_state': link_state,
            'first': first,
            'second': second,
            'state_dim': state_dim
        })[0]
        loss = F.mse_loss(ret, value)

        return loss

    def update(self, actions, actions_probs, tensors, critic_features, returns, advantages):

        for pos in range(self.buffer_size):
            tensor = tensors[pos]
            critic_feature = critic_features[pos]
            action = actions[pos]
            ret = returns[pos]
            adv = advantages[pos]
            action_dist = actions_probs[pos]

            update_tensor = {
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'],
                'first': tensor['first'],
                'second': tensor['second'],
                'state_dim': tensor['state_dim'],
                'num_actions': tensor['num_actions'],
                'link_state_critic': critic_feature['link_state'],
                'old_act': action,
                'adv': adv,
                'old_policy_probs': action_dist,
                'first_critic': critic_feature['first'],
                'second_critic': critic_feature['second'],
                'ret': ret,
            }

            self.buffer.append(update_tensor)

        for i in range(self.update_times):
            np.random.shuffle(self.buffer_index)
            for start in range(0, self.buffer_size, self.mini_batch):
                end = start + self.mini_batch
                entropy = 0
                actor_loss = 0
                critic_loss = 0
                for index in self.buffer_index[start:end]:
                    sample = self.buffer[index]

                    sample_actor_loss, sample_entropy = self._compute_actor_loss(sample['adv'], sample['old_act'],
                                                                                 sample['old_policy_probs'],
                                                                                 sample['link_state'],
                                                                                 sample['graph_id'], sample['first'],
                                                                                 sample['second'], sample['state_dim'],
                                                                                 sample['num_actions'])
                    sample_critic_loss = self._compute_critic_loss(sample['ret'], sample['link_state_critic'],
                                                                   sample['first_critic'], sample['second_critic'],
                                                                   sample['state_dim'])
                    entropy += sample_entropy
                    actor_loss += sample_actor_loss
                    critic_loss += sample_critic_loss

                entropy /= self.mini_batch
                actor_loss = actor_loss / self.mini_batch - self.entropy_beta * entropy
                critic_loss /= self.mini_batch

                total_loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        self.buffer.clear()
        gc.collect()
        return actor_loss, critic_loss


