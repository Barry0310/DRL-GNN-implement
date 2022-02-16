from gnn_model import MPNN

data = {
    'link_state': [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
    'pair': [[0, 0, 0], [1, 1, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1], [0, 0, 0], [2, 2, 2], [0, 0, 0]],
}

model = MPNN(3, 4)

output = model(data)

print(output.item())

import networkx as nx

G = nx.Graph()

G.add_node(0)
G.add_node(1)
print(G.nodes())
G.add_edge(0, 1)
G.add_edge(1, 0)
print(G.edges())
print(G.degree[1])