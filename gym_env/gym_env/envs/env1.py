import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json
import gc


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

def generate_nx_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_geant2_graph()
    else:
        G = create_gbn_graph()

    # nx.draw(G, with_labels=True)
    # plt.show()
    # plt.clf()

    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.
    for i, j in G.edges():
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    return G

class Env1(gym.Env):
    def __init__(self, topology, demand_list):
        self.graph = generate_nx_graph(topology)
        self.demand_list = demand_list

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
