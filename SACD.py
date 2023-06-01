import torch
import torch.nn.functional as F
import torch.optim as optim
from Actor import Actor
from Critic import Critic
import numpy as np
import copy
from collections import deque
import random


class SACD:
    def __init__(self, hp, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = hp['gamma']
        self.tau = 0.005
        self.feature_size = hp['feature_size']
        self.batch_size = hp['batch_size']

        self.alpha = hp['alpha']
        self.target_entropy = 0.5 * (-np.log(1 / hp['avg_a_dim']))  # H(discrete)>0
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=hp['a_lr'])

        self.actor = Actor(feature_size=self.feature_size, t=hp['t'], readout_units=hp['readout_units']).to(self.device)
        self.a_optimizer = optim.AdamW(self.actor.parameters(), lr=hp['a_lr'])

        self.critic = Critic(feature_size=self.feature_size, t=hp['t'], readout_units=hp['readout_units']).to(self.device)
        self.c_optimizer = optim.AdamW(self.critic.parameters(), lr=hp['c_lr'])
        self.critic_target = copy.deepcopy(self.critic)

        self.replay_buffer = deque(maxlen=hp['buffer_size'])

        self.K_path = 0

    def add_exp(self, env, st, src, dst, demand, action, reward, done):
        stp1 = self.input_transform(env, src, dst, demand)
        exp = {
            'st': st,
            'stp1': stp1,
            'a': action,
            'r': reward,
            'done': done
        }
        self.replay_buffer.append(exp)

    def sample(self):
        batch_data = random.sample(self.replay_buffer, self.batch_size)

        return batch_data

    def predict(self, env, src, dst, demand):
        tensor = self.input_transform(env, src, dst, demand)
        probs = self.actor(tensor)

        return probs, tensor

    def get_path(self, env, src, dst):
        path = []
        current_path = env.shortest_paths[src, dst]
        i = 0
        j = 1
        while j < len(current_path):
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

    def get_path_features(self, env, src, dst, mid, demand):
        temp = {
            'path': [],
            'demand': [demand],
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

        path_feature = {'path': temp['path'], 'path_state': path_state}

        return path_feature

    def input_transform(self, env, src, dst, demand):
        list_k_features = []

        middle_point_list = []

        if self.K_path > 0:
            middle_point_list = range(self.K_path)
        else:
            middle_point_list = env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]

        for mid in range(len(middle_point_list)):
            features = self.get_path_features(env, src, dst, middle_point_list[mid], demand)
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
        return tensor

    def train(self):
        batch_data = self.sample()

        # ------------------------------------------ Train Critic ----------------------------------------#
        q1_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        q2_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for i in range(self.batch_size):
            '''Compute the target soft Q value'''
            with torch.no_grad():
                next_probs = self.actor(batch_data[i]['stp1'])
                next_log_probs = torch.log(next_probs + 1e-8)
                next_q1, next_q2 = self.critic_target(batch_data[i]['stp1'])
                min_next_q = torch.min(next_q1, next_q2)
                v_next = torch.sum(next_probs * (min_next_q - self.alpha * next_log_probs))
                target_q = batch_data[i]['r'] + (1-batch_data[i]['done']) * self.gamma * v_next

            '''Update soft Q net'''
            q1_all, q2_all = self.critic(batch_data[i]['st'])
            q1, q2 = q1_all[batch_data[i]['a']], q2_all[batch_data[i]['a']]
            q1_loss += F.mse_loss(q1, target_q)
            q2_loss += F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        q_loss /= self.batch_size
        self.c_optimizer.zero_grad()
        q_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
        self.c_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        for params in self.critic.parameters():
            # Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
            params.requires_grad = False

        a_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        entropy = torch.tensor(0, dtype=torch.float32, device=self.device)
        for i in range(self.batch_size):
            probs = self.actor(batch_data[i]['st'])
            log_probs = torch.log(probs + 1e-8)
            entropy += -torch.sum(probs * log_probs)
            with torch.no_grad():
                q1_all, q2_all = self.critic(batch_data[i]['st'])
            min_q_all = torch.min(q1_all, q2_all)
            a_loss += torch.sum(probs * (self.alpha * log_probs - min_q_all))
        a_loss /= self.batch_size
        self.a_optimizer.zero_grad()
        a_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
        self.a_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = True

        # ------------------------------------------ Train Alpha ----------------------------------------#
        with torch.no_grad():
            H_mean = entropy/self.batch_size
        alpha_loss = self.log_alpha * (H_mean - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ----------------------------------#
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return a_loss.cpu().detach().numpy(), q_loss.cpu().detach().numpy()
