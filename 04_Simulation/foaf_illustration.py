import matplotlib.pyplot as plt
import networkx as nx

n = nx.Graph()
n.add_edge( 'Self', 'Friend1')
n.add_edge('Self', 'Friend2')
n.add_edge('Friend1', 'A')
n.add_edge('Friend1', 'B')
n.add_edge('Friend1', 'C')
n.add_edge('Friend2', 'X')
n.add_edge('Friend2', 'Y')
n.add_edge('Friend2', 'Z')
n.add_edge('Friend1', 'D')
n.add_edge('Friend2', 'D')
n.add_edge('C', 'X')


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
# pos = {'Alice':[1,0],'Bob':[0,0],'Charli':[0,1],'David':[1,1],'Emma':[2,1],'Frank':[2,0]}

ce = [('Alice', 'Bob'), ('Bob', 'Charli'), ('Charli', 'David'), ('David', 'Alice')]
for u,v in n.edges:
    if u in ('Friend1', 'Friend2'):
        n[u][v]['color'] = 'g'
    else:
        n[u][v]['color'] = 'k'
colors = [n[u][v]['color'] for u,v in n.edges]

plt.figure()
pos = nx.spring_layout(n)
nx.draw(n, pos, edge_color=colors, width=1, linewidths=1, node_size=500,\
    node_color='pink', alpha=0.9, labels={node:node for node in n.nodes()},\
    connectionstyle='arc3, rad = 0.15')

# nx.draw_networkx_edge_labels(n.G, pos, edge_labels=balances,font_color='k', label_pos=0.8)
plt.axis('off')
plt.savefig('foaf.pdf')
