import numpy as np
import torch
from Actor import Actor
from Critic import Critic
import torch.optim as optim
from collections import deque
import gc


class PPOAC:
    def __init__(self, hyper_parameter, device=None):
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
        self.optimizer = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=H['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=H['lr_decay_step'],
                                                   gamma=H['lr_decay_rate'])

        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_index = np.arange(self.buffer_size)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.K_path = 0

    def predict(self, env, src, dst):
        list_k_features = []
        middle_point_list = []
        
        if self.K_path > 0:
            middle_point_list = range(self.K_path)
        else:
            middle_point_list = env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]

        for mid in range(len(middle_point_list)):
            features = self.actor_get_graph_features(env, src, dst, middle_point_list[mid])
            list_k_features.append(features)

        path_id = []
        sequence = []
        link_id = []
        for i in range(len(list_k_features)):
            for j in range(len(list_k_features[i]['path'])):
                path_id.append(i)
                sequence.append(j)
            link_id = link_id + list_k_features[i]['path']

        temp = {
            'num_edges': env.numEdges,
            'capacity': env.link_capacity_feature,
            'utilization': np.divide(env.edge_state[:, 0], env.edge_state[:, 1]),
        }
        temp['utilization'] = torch.reshape(torch.tensor(temp['utilization'][0:temp['num_edges']], dtype=torch.float32),
                                            (temp['num_edges'], 1))
        temp['capacity'] = torch.reshape(torch.tensor(temp['capacity'][0:temp['num_edges']], dtype=torch.float32),
                                         (temp['num_edges'], 1))
        hidden_states = torch.cat([temp['utilization'], temp['capacity']], dim=1)
        link_state = torch.nn.functional.pad(hidden_states, (0, self.feature_size - 2), 'constant')

        path_state = torch.stack([v['path_state'] for v in list_k_features], dim=0)

        tensor = {
            'link_state': link_state.to(self.device),
            'path_state': path_state.to(self.device),
            'path_id': torch.tensor(path_id).to(self.device),
            'sequence': torch.tensor(sequence).to(self.device),
            'link_id': torch.tensor(link_id).to(self.device),
            'num_actions': len(middle_point_list),
        }
        q_values = self.actor(tensor)
        q_values = torch.reshape(q_values, (-1, ))
        soft_max_q_values = torch.nn.functional.softmax(q_values, dim=0)

        return soft_max_q_values, tensor

    def get_path(self, env, src, dst):
        path = []
        current_path = env.shortest_paths[src, dst]
        i = 0
        j = 1
        while (j < len(current_path)):
            path.append(env.edgesDict[str(current_path[i]) + ':' + str(current_path[j])])
            i = i + 1
            j = j + 1
        return path
        
    def get_path_K(self, env, src, dst, k):
        path = []
        current_path = env.allPaths[str(src)+':'+str(dst)][k]
        i = 0
        j = 1
        while j < len(current_path):
            path.append(env.edgesDict[str(current_path[i]) + ':' + str(current_path[j])])
            i = i + 1
            j = j + 1
        return path

    def actor_get_graph_features(self, env, src, dst, mid):
        temp = {
            'path': [],
            'demand': [env.TM[src][dst]],
            'link_capacity': env.edge_state[:, 1],
        }

        if self.K_path > 0:
            temp['path'] = self.get_path_K(env, src, dst, mid)
        else:
            temp['path'] = self.get_path(env, src, mid)
            if mid != dst:
                temp['path'] = temp['path'] + self.get_path(env, mid, dst)

        temp['demand'][0] = temp['demand'][0] / min(temp['link_capacity'][temp['path']])

        path_state = torch.nn.functional.pad(torch.tensor(temp['demand'], dtype=torch.float32),
                                             (0, self.feature_size - 1), 'constant')

        inputs = {'path': temp['path'], 'path_state': path_state}

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

        temp['utilization'] = torch.reshape(torch.tensor(temp['utilization'][0:temp['num_edges']], dtype=torch.float32),
                                            [temp['num_edges'], 1])
        temp['capacity'] = torch.reshape(torch.tensor(temp['capacity'][0:temp['num_edges']], dtype=torch.float32),
                                         [temp['num_edges'], 1])

        hidden_states = torch.cat([temp['utilization'], temp['capacity']], dim=1)
        link_state = torch.nn.functional.pad(hidden_states, (0, self.feature_size - 2), 'constant')

        inputs = {'link_state': link_state.to(self.device),
                  'first': torch.tensor(temp['first'][0:temp['length']]).to(self.device),
                  'second': torch.tensor(temp['second'][0:temp['length']]).to(self.device),
                  'state_dim': self.feature_size}

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

    def _compute_actor_loss(self, adv, old_act, old_policy_probs, actor_feature):
        old_policy_probs = old_policy_probs.detach()

        q_values = self.actor(actor_feature)
        q_values = torch.reshape(q_values, (-1,))
        new_policy_probs = torch.nn.functional.softmax(q_values, dim=0)

        ratio = torch.exp(
            torch.log(torch.sum(old_act * new_policy_probs)) - torch.log(torch.sum(old_act * old_policy_probs))
        )
        surr1 = -ratio*adv
        surr2 = -torch.clip(ratio, min=1-0.1, max=1+0.1) * adv

        loss = torch.max(surr1, surr2)
        entropy = -torch.sum(torch.log(new_policy_probs) * new_policy_probs)

        return loss, entropy

    def _compute_critic_loss(self, ret, critic_feature):

        value = self.critic(critic_feature)[0]
        loss = torch.square(ret - value)

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
                'actor_feature': tensor,
                'critic_feature': critic_feature,
                'old_act': action.to(self.device),
                'adv': adv,
                'old_policy_probs': action_dist,
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
                                                                                 sample['actor_feature'])
                    sample_critic_loss = self._compute_critic_loss(sample['ret'], sample['critic_feature'])
                    entropy += sample_entropy
                    actor_loss += sample_actor_loss
                    critic_loss += sample_critic_loss

                entropy /= self.mini_batch
                actor_loss = actor_loss / self.mini_batch - self.entropy_beta * entropy
                critic_loss /= self.mini_batch

                total_loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters())+list(self.critic.parameters()),
                                               max_norm=self.clip_value)
                self.optimizer.step()

        self.buffer.clear()
        gc.collect()
        return actor_loss, critic_loss


