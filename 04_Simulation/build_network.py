from Network import Network
import time
import networkx as nx

channel_file = 'channels.json'

start = time.time()
n = Network.parse_clightning(channel_file)
end = time.time()
G = n.G
print(end - start, "time to read-in network")
assert True == isinstance(n.G, nx.DiGraph), 'Network is not as exptected'