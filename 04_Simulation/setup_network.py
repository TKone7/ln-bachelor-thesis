import csv
import json
from operator import itemgetter
import networkx as nx

# with open('dump.csv', 'r') as networkcsv:  # Open the file
#     reader = csv.reader(networkcsv, delimiter='\t')  # Read the csv
#     rows = [n for n in reader]
#     headers = rows[0:1]
#     channels = rows[1:]

with open('dump.json', 'r') as network:  # Open the file
    channels = json.load(network)

nodes = set([c['source'] for c in channels])
nodes.update(set([c['destination'] for c in channels]))

edges = [tuple([c['source'], c['destination'], {'chan_id': c['short_channel_id'], 'cap': c['satoshis'], 'base_fee_msat': c['base_fee_millisatoshi'], 'fee_per_millionth': c['fee_per_millionth']}]) for c in channels]
#edges = [tuple([c['source'], c['destination']]) for c in channels]

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

[print(e) for e in G.edges(data=True) if e[2]['chan_id']=='570629x1280x1']

print(nx.info(G, satoshilabs))



G = nx.MultiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
print(nx.info(G))
[print(e) for e in G.edges(data=True) if e[0]=='02f2db91d9c63aeeff2b2661b5398e4146aeb2cdb10fa48e570a2c20a420072672' and e[1] == '0331f80652fb840239df8dc99205792bba2e559a05469915804c08420230e23c7c']

# why not multi?




