import matplotlib.pyplot as plt
import networkx as nx
edges = [['Alice', 'Bob'], ['Bob', 'Alice']]
G = nx.DiGraph()
G.add_edges_from(edges)
# pos = nx.bipartite_layout(G)
pos = {'Alice':[0,0],'Bob':[1,0]}
# pos2 = {'Alice':[0,0],'Bob':[1,0]}

plt.figure()
nx.draw(G, pos, edge_color='black', width=1, linewidths=1, node_size=500,\
    node_color='pink', alpha=0.9, labels={node:node for node in G.nodes()},\
    connectionstyle='arc3, rad = 0.15')
nx.draw_networkx_edge_labels(G, pos, edge_labels={('Alice','Bob'):'0.2 BTC',\
('Bob','Alice'):'0.8 BTC'},font_color='red', label_pos=0.8)

plt.axis('off')
plt.savefig('simple_channel.pdf')
