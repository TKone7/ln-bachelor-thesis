import numpy as np
import scipy.stats
from Network import Network
import networkx as nx

n = Network.restore_snapshot('3a65a961')

balances_dict = dict()
for e in n.G.edges(data=True):
    bal = e[2]['balance']
    if e[0] not in balances_dict.keys():
        balances_dict[e[0]] = bal
    else:
        balances_dict[e[0]] += bal

degree_centrality_dict = nx.degree_centrality(n.G)
betweenness_dict = nx.betweenness_centrality(n.G)
eigenvector_dict = nx.eigenvector_centrality(n.G)
pagerank_dict = nx.pagerank(n.G)
# create arrays from dictionary
degree_centrality = np.array([degree_centrality_dict[node] for node in n.G.nodes])
betweenness = np.array([betweenness_dict[node] for node in n.G.nodes])
eigenvector = np.array([eigenvector_dict[node] for node in n.G.nodes])
pagerank = np.array([pagerank_dict[node] for node in n.G.nodes])
balance = np.array([balances_dict[node] for node in n.G.nodes])
# combine the arrays
combined = np.array([degree_centrality, betweenness, eigenvector, pagerank, balance])

rho, pval = scipy.stats.spearmanr(combined, axis=1)

print(rho)