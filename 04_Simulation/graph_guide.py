import networkx as nx
import matplotlib.pyplot as plt
# G_symmetric = nx.Graph()
# G_symmetric.add_edge('Amitabh Bachchan','Abhishek Bachchan')
# G_symmetric.add_edge('Amitabh Bachchan','Aamir Khan')
# G_symmetric.add_edge('Amitabh Bachchan','Akshay Kumar')
# G_symmetric.add_edge('Amitabh Bachchan','Dev Anand')
# G_symmetric.add_edge('Abhishek Bachchan','Aamir Khan')
# G_symmetric.add_edge('Abhishek Bachchan','Akshay Kumar')
# G_symmetric.add_edge('Abhishek Bachchan','Dev Anand')
# G_symmetric.add_edge('Dev Anand','Aamir Khan')
#
# nx.draw_networkx(G_symmetric)


# G_asymmetric = nx.DiGraph()
# G_asymmetric.add_edge('A','B')
# G_asymmetric.add_edge('A','D')
# G_asymmetric.add_edge('C','A')
# G_asymmetric.add_edge('D','E')
# nx.spring_layout(G_asymmetric)
# nx.draw_networkx(G_asymmetric)
G_weighted = nx.Graph()
G_weighted.add_edge('Amitabh Bachchan','Abhishek Bachchan', weight=205)
G_weighted.add_edge('Amitabh Bachchan','Aaamir Khan', weight=8)
G_weighted.add_edge('Amitabh Bachchan','Akshay Kumar', weight=11)
G_weighted.add_edge('Amitabh Bachchan','Dev Anand', weight=1)
G_weighted.add_edge('Abhishek Bachchan','Aaamir Khan', weight=4)
G_weighted.add_edge('Abhishek Bachchan','Akshay Kumar',weight=7)
G_weighted.add_edge('Abhishek Bachchan','Dev Anand', weight=1)
G_weighted.add_edge('Dev Anand','Aaamir Khan',weight=1)
nx.draw_networkx(G_weighted)

plt.show()

# Degree of a node defines the number of connections a node has.
nx.degree(G_weighted, 'Dev Anand')
# We can also determine the shortest path between two nodes and its length
nx.shortest_path(G_weighted, 'Dev Anand', 'Akshay Kumar')
nx.shortest_path_length(G_weighted, 'Dev Anand', 'Akshay Kumar')

# Eccentricity of a node A is defined as the largest distance between A and all other nodes
nx.eccentricity(G_weighted,'Abhishek Bachchan')

# centrality measures
nx.degree_centrality(G_weighted)
cent = nx.eigenvector_centrality(G_weighted)
nx.betweenness_centrality(G_weighted)