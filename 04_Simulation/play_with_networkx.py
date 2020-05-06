import json
import networkx as nx
from random import choice


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
acinq = '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'
candle = '02f31ff9c53e1773431f248ea81b97f09f98bb8798747e67e9f080d1d20b7d644d'
ownbank = '021ed3e1d88afbfa41d70531bd83c8d5d8361ac354800daf66ae17ddb1d94c68d4'
savesats = '0266ac6fc120de46e2f84a6cccdbe7b6e79199a4664c44539d0c723eb34b7d5261'
in_edges = [e for e in G.in_edges(satoshilabs,data=True)]
out_edges = [e for e in G.out_edges(satoshilabs,data=True)]

for e in out_edges:
    if not e[2]['chan_id'] in [ed[2]['chan_id'] for ed in in_edges]:
        print('not available', e[1], e[2])
    else:
        print('available', e[1], e[2])

[print(e) for e in G.edges(data=True) if e[2]['chan_id']=='570629x1280x1']

is_simple = nx.is_simple_path(G, [acinq, satoshilabs])

n1 = choice(list(G.nodes()))
n2 = choice(list(G.nodes()))
path = nx.shortest_path(G,n1, n2)
print(path)
for i in range(0,len(path)-1):
    print(G[path[i]][path[i+1]])

G.in_edges(n1)
G.out_edges(n1)

len(G.edges)
apsp = dict(nx.all_pairs_shortest_path(G, 8))
path = apsp[n1][n2]
for i in range(0,len(path)-1):
    print(G[path[i]][path[i+1]])

for x in G:
    for y in G[x]:
        print(G[x][y])

map = {}
map['abc'] = 'here'
map['def'] = 'we'
map['ghi'] = 'go'
for k,v in map.items():
    print(k,v)
list(nx.simple_cycles(G))
cycles = list(nx.find_cycle(G, acinq))

soc = None
soc = nx.MultiDiGraph()
soc.add_nodes_from(['A', 'B', 'C'])
soc.add_edges_from([('A','B',{'id': 'ab1'}),('A','B',{'id': 'ab2'}), ('B','A'), ('B','C'), ('C','B'), ('A','C'), ('C','A')])
l = list(nx.all_simple_paths(soc, 'A', 'B'))