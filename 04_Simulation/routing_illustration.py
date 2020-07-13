import matplotlib.pyplot as plt
import networkx as nx
from Network import Network

n = Network.restore_snapshot('83ab53d1')

balances = dict()
for edge in n.G.edges(data=True):
    balances[(edge[0],edge[1])] = edge[2]['balance']

# pos = dict()
# nodes = list(n.G.nodes)
# sorted(nodes)
# for e, node in enumerate(nodes):
#     x = e % 2
#     y = int(e / 2)
#     pos[node] = [x,y]
# edges = [['Alice', 'Bob'], ['Bob', 'Alice']]
# G = nx.DiGraph()
# G.add_edges_from(edges)
# # pos = nx.bipartite_layout(G)
pos = {'Alice':[1,0],'Bob':[0,0],'Charli':[0,1],'David':[1,1],'Emma':[2,1],'Frank':[2,0]}

ce = [('Alice', 'Bob'), ('Bob', 'Charli'), ('Charli', 'David'), ('David', 'Alice')]
for u,v in n.G.edges:
    if (u,v) in ce:
        n.G[u][v]['color'] = 'r'
    else:
        n.G[u][v]['color'] = 'k'
colors = [n.G[u][v]['color'] for u,v in n.G.edges]

plt.figure()
nx.draw(n.G, pos, edge_color='k', width=1, linewidths=1, node_size=500,\
    node_color='pink', alpha=0.9, labels={node:node for node in n.G.nodes()},\
    connectionstyle='arc3, rad = 0.15')
nx.draw_networkx_edge_labels(n.G, pos, edge_labels=balances,font_color='k', label_pos=0.8)
plt.axis('off')
plt.savefig('rebalancing.pdf')

# change balances
new_balances = dict()
for tx in ce:
    n.G[tx[0]][tx[1]]['balance'] -= 40
    n.G[tx[1]][tx[0]]['balance'] += 40
    old = balances[(tx[0],tx[1])]
    change = ' - 40 = '
    new = old -40
    new_balances[(tx[0],tx[1])] = str(old) + change + str(new)
    old = balances[(tx[1],tx[0])]
    change = ' + 40 = '
    new = old + 40
    new_balances[(tx[1],tx[0])] = str(old) + change + str(new)

balances = dict()
for edge in n.G.edges(data=True):
    balances[(edge[0],edge[1])] = edge[2]['balance']

plt.figure()
nx.draw(n.G, pos, edge_color=colors, width=1, linewidths=1, node_size=500,\
    node_color='pink', alpha=0.9, labels={node:node for node in n.G.nodes()},\
    connectionstyle='arc3, rad = 0.15')
nx.draw_networkx_edge_labels(n.G, pos, edge_labels=balances,font_color='red', label_pos=0.8)
plt.axis('off')
plt.savefig('rebalancing2.pdf')
