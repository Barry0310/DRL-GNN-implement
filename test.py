import numpy as np

from gnn_model import MPNN
import matplotlib.pyplot as plt

'''
data = {
    'link_state': [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
    'pair': [[0, 0, 0], [1, 1, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1], [0, 0, 0], [2, 2, 2], [0, 0, 0]],
}

model = MPNN(3, 4)

output = model(data)

print(output.item())


import networkx as nx
import numpy as np

Gbase = nx.Graph()
Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
Gbase.add_edges_from(
    [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
     (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
     (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

print(Gbase)
ordered_edges = sorted([edge for edge in Gbase.edges()])
numEdges = len(Gbase.edges())
graph_state = np.zeros((numEdges, 2)) # DRL stata
print('number of edges', numEdges)
edgesDict = dict() # 對應link及link編號


idx = 0
for edge in Gbase.edges():
    Gbase.get_edge_data(*edge)['capacity'] = 200
    edgesDict[edge] = idx
    neighbour_edges = list(Gbase.edges(edge[0])) + list(Gbase.edges(edge[1]))
    graph_state[idx][0] = Gbase.get_edge_data(*edge)["capacity"]
    idx = idx + 1
pos = nx.spring_layout(Gbase)
edge_labels = nx.get_edge_attributes(Gbase, 'capacity')
nx.draw(Gbase, pos, with_labels=True)
nx.draw_networkx_edge_labels(Gbase, pos, edge_labels=edge_labels)
plt.show()
plt.clf()
neighbour_edges = dict()
for n1, n2 in ordered_edges:
    neighbour_edges[(n1, n2)] = list()
    for m, n in list(Gbase.edges(n1))+list(Gbase.edges(n2)):
        if (n1!=m or n2!=n) and (n1!=n or n2!=m):
            neighbour_edges[(n1, n2)].append((m, n))
#print(neighbour_edges)

shortest_path = dict(nx.all_pairs_shortest_path(Gbase))
action_space = dict()
for i in Gbase.nodes():
    for j in Gbase.nodes():
        if i == j:
            continue
        action_space[(i, j)] = []
        for k in Gbase.nodes():
            if k == i or k == j:
                continue
            if j not in shortest_path[i][k] and i not in shortest_path[k][j]:
                action_space[(i, j)].append(k)


print(shortest_path)
print(action_space)
print([len(action_space[i]) for i in action_space])
for i in Gbase.nodes():
    if i == 0 or i == 2:
        continue
    if i in shortest_path[0][i] or i in shortest_path[i][2]:
        print('middle: ', i)
        print(shortest_path[0][i], shortest_path[i][2])

if Gbase[0][2] == Gbase[2][0]:
    Gbase[0][2]['capacity'] = 190
    print(Gbase[0][2])
else:
    print('bad')
print(Gbase[2][0])

a = [0, 1, 2]
b = a
print(a[0:-1]+b)
'''
import gym
import gym_env
ENV_NAME = 'GraphEnv-v1'
env_training = gym.make(ENV_NAME)
state, demand = env_training.reset(topology=0, demand_list=[(0, 2, 100)])
print(demand)
for i in env_training.neighbor_edges:
    for j in env_training.neighbor_edges[i]:
        print(np.concatenate((state[env_training.edges_dict[i]], state[env_training.edges_dict[j]]), axis=0))
    break
print(env_training.mark_action(1))


