import csv
from operator import itemgetter
import networkx as nx

with open('dump.csv', 'r') as networkcsv:  # Open the file
    reader = csv.reader(networkcsv, delimiter='\t')  # Read the csv
    rows = [n for n in reader]
    headers = rows[0:1]
    channels = rows[1:]

nodes = set([c[0] for c in channels])
nodes.update(set([c[1] for c in channels]))

edges = [tuple([c[0], c[1], {'chan_id': c[2], 'cap': c[3], 'base_fee_msat': c[4], 'fee_per_millionth': c[5]}]) for c in channels]

print(len(nodes))
print(len(edges))

G = nx.MultiDiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
print(nx.info(G))

satoshilabs = '0279c22ed7a068d10dc1a38ae66d2d6461e269226c60258c021b1ddcdfe4b00bc4'
in_edges = [e for e in G.in_edges(satoshilabs,data=True)]
out_edges = [e for e in G.out_edges(satoshilabs,data=True)]

for e in out_edges:
    if not e[2]['chan_id'] in [ed[2]['chan_id'] for ed in in_edges]:
        print('not available', e[1], e[2])
    else:
        print('available', e[1], e[2])

[print(e) for e in G.edges(data=True) if e[2]['chan_id']=='539637x1777x0']

print(nx.info(G, satoshilabs))


