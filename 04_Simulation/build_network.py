from Network import Network
import time, json
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import pylab
def random_node():
    return random.choice(list(n.G.nodes))

channel_file = 'channels.json'
snapshot = 'f1f1896f' #  f1f1896f for testing network use '3bf6a9dd' or f680d907
start = time.time()
n = None
# n = Network.parse_clightning('channels.json') #new: 0f4575e4 older: f1f1896f
n = Network.restore_snapshot(snapshot)
n.create_snapshot()
end = time.time()
print(end - start, "time to read-in network")

start = time.time()
n.compute_rebalance_network()
end = time.time()
print(end - start, "time to cal rebalance directions")

start = time.time()
n.compute_circles()
end = time.time()
print(end - start, "time to calc operations")

start = time.time()
n.rebalance()
end = time.time()
print(end - start, "time to rebalance operations")
