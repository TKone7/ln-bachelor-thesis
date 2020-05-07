from Network import Network
import time
import networkx as nx

channel_file = 'channels.json'

start = time.time()
t = Network.parse_clightning(channel_file)
t2 = Network.parse_clightning(channel_file)

n = Network.restore_snapshot('directed_lightning_network_rp')
end = time.time()
# G = n.G
print(end - start, "time to read-in network")
assert True == isinstance(n.G, nx.DiGraph), 'Network is not as exptected'

# print(nx.info(G))
print('network1 has {} nodes an {} channels'.format(len(t.G), len(t.G.edges)))
print('network1 has {} nodes an {} channels'.format(len(t2.G), len(t2.G.edges)))