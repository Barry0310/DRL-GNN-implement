import gym
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import copy


def create_geant2_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])

    return Gbase


def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase


def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    return Gbase


def generate_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_geant2_graph()
    else:
        G = create_gbn_graph()

    idx = 0
    for i, j in G.edges():
        G.get_edge_data(i, j)['capacity'] = 200
        G.get_edge_data(i, j)['utilization'] = 0
        G.get_edge_data(i, j)['bwAlloc'] = 0
        idx = idx + 1

    return G


class Env1(gym.Env):
    def __init__(self):
        self.edges_dict = None  # 對應 link 及 link 編號
        self.neighbor_edges = None  # 紀錄臨邊資訊供 gnn 使用
        self._graph = None
        self._demand_list = None
        self._demand_routing = None  # 紀錄 demand 路由路徑方便 step 更新
        self._num_edges = None
        self._ordered_edges = None
        self._graph_state = None  # DRL stata
        self._shortest_path = None  # 儲存所有 node pair的最短路
        self._demand_idx = None  # 目前待處理的 demand
        self._last_max_util = None  # 紀錄上一步最大利用率方便下一步計算 reward
        self._done = None  # 是否完成episode

    def _max_link_util(self):
        """
        find link have maximum link utilization
        """
        max_util = 0
        for i in self._graph.edges():
            if self._graph.get_edge_data(*i)['utilization'] > max_util:
                max_util = self._graph.get_edge_data(*i)['utilization']
        return max_util

    def mark_action(self, action):
         """
        mark action on links the path have
        """
        marked = copy.deepcopy(self._graph_state)
        if action == -1:
            return marked
        demand = self._demand_list[self._demand_idx]
        temp = self._shortest_path[demand[0]][action]
        for i in range(len(temp) - 1):
            marked[self.edges_dict[(temp[i], temp[i+1])]][2] = demand[2]
        temp = self._shortest_path[action][demand[1]]
        for i in range(len(temp) - 1):
            marked[self.edges_dict[(temp[i], temp[i + 1])]][2] = demand[2]
        return marked

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def step(self, action):
        if action != 0:
            demand = self._demand_list[self._demand_idx]
            action = self.action_space[self.edges_dict[(demand[0], demand[1])]][action]
            temp = self._shortest_path[demand[0]][action]
            for i in range(len(temp) - 1):
                self._graph[temp[i]][temp[i + 1]]['bwAlloc'] += demand[2]
                self._graph[temp[i]][temp[i + 1]]['utilization'] = self._graph[temp[i]][temp[i + 1]]['bwAlloc'] \
                                                                   / self._graph[temp[i]][temp[i + 1]]['capacity']
                self._graph_state[self.edges_dict[(temp[i], temp[i + 1])]][1] = self._graph[temp[i]][temp[i + 1]]['utilization']

            temp = self._shortest_path[action][demand[1]]
            for i in range(len(temp) - 1):
                self._graph[temp[i]][temp[i + 1]]['bwAlloc'] += demand[2]
                self._graph[temp[i]][temp[i + 1]]['utilization'] = self._graph[temp[i]][temp[i + 1]]['bwAlloc'] \
                                                                   / self._graph[temp[i]][temp[i + 1]]['capacity']
                self._graph_state[self.edges_dict[(temp[i], temp[i + 1])]][1] = self._graph[temp[i]][temp[i + 1]]['utilization']

            temp = self._demand_routing[demand]
            for i in range(len(temp) - 1):
                self._graph[temp[i]][temp[i + 1]]['bwAlloc'] -= demand[2]
                self._graph[temp[i]][temp[i + 1]]['utilization'] = self._graph[temp[i]][temp[i + 1]]['bwAlloc'] \
                                                                   / self._graph[temp[i]][temp[i + 1]]['capacity']
                self._graph_state[self.edges_dict[(temp[i], temp[i + 1])]][1] = self._graph[temp[i]][temp[i + 1]]['utilization']
            self._demand_routing[demand] = self._shortest_path[demand[0]][action][0:-1] + self._shortest_path[action][demand[1]]

        max_util = self._max_link_util()
        reward = self._last_max_util - max_util
        self._last_max_util = max_util

        self._demand_idx = self._demand_idx + 1
        if self._demand_idx == len(self._demand_list):
            self._done = True
            demand = None
        else:
            self._done = False
            demand = self._demand_list[self._demand_idx]

        return copy.deepcopy(self._graph_state), self._done, demand, reward

    def reset(self, topology, demand_list):
        self._graph = generate_graph(topology)
        self._demand_list = demand_list
        self._demand_idx = 0
        self._num_edges = len(self._graph.edges())
        self._ordered_edges = sorted([edge for edge in self._graph.edges()])
        self.edges_dict = dict()
        self._graph_state = np.zeros((self._num_edges, 3))
        self._last_max_util = 0
        self._done = False

        idx = 0
        for n1, n2 in self._ordered_edges:
            self.edges_dict[(n1, n2)] = idx
            self.edges_dict[(n2, n1)] = idx
            self._graph_state[idx][0] = self._graph.get_edge_data(n1, n2)['capacity']
            self._graph_state[idx][1] = self._graph.get_edge_data(n1, n2)['utilization']
            idx = idx + 1

        self.neighbor_edges = dict()
        for n1, n2 in self._ordered_edges:
            self.neighbor_edges[(n1, n2)] = list()
            for m, n in list(self._graph.edges(n1)) + list(self._graph.edges(n2)):
                if (n1 != m or n2 != n) and (n1 != n or n2 != m):
                    self.neighbor_edges[(n1, n2)].append((m, n))

        self._shortest_path = dict(nx.all_pairs_shortest_path(self._graph))
        self.action_space = dict()
        for i in self._graph.edges():
            self.action_space[self.edges_dict[i]] = [-1]
            for k in self._graph.nodes():
                if k == i[0] or k == i[1]:
                    continue
                if i[1] not in self._shortest_path[i[0]][k] or i[0] not in self._shortest_path[k][i[1]]:
                    self.action_space[self.edges_dict[i]].append(k)

        self._demand_routing = dict()
        for i in self._demand_list:
            temp = self._shortest_path[i[0]][i[1]]
            for j in range(len(temp) - 1):
                self._graph[temp[j]][temp[j - 1]]['bwAlloc'] += i[2]
                self._graph[temp[j]][temp[j - 1]]['utilization'] = self._graph[temp[j]][temp[j - 1]]['bwAlloc'] \
                                                                   / self._graph[temp[j]][temp[j - 1]]['capacity']
                self._graph_state[self.edges_dict[(temp[j], temp[j - 1])]][1] \
                    = self._graph[temp[j]][temp[j - 1]]['utilization']
            self._demand_routing[i] = self._shortest_path[i[0]][i[1]]
        self._last_max_util = self._max_link_util()

        return copy.deepcopy(self._graph_state), self._demand_list[self._demand_idx]

    def render(self, mode='human'):
        if mode == 'human':
            pos = nx.spring_layout(self._graph)
            edge_labels = nx.get_edge_attributes(self._graph, 'capacity')
            nx.draw(self._graph, pos, with_labels=True)
            nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels)
            plt.show()
            plt.clf()
