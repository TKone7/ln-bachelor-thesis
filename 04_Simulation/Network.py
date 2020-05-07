import random
import json, logging, hashlib
import networkx as nx
import numpy as np

FORMAT ='%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format = FORMAT)
logger = logging.getLogger(__name__)
class Network:
    def __init__(self, G):
        self.G = G
        self.excluded = set()
        self.__history = []
        # calculate initial gini coefficients for all nodes
        self.__update_ginis()
        self.flow = None

        self.fingerprint = self.__fingerprint()
        logger.info('Fingerprint is {}'.format(self.fingerprint))

    def __fingerprint(self):
        # calculate a fingerprint for the initial network state. Should include:
        # - nodes / edges / attributes
        # - excluded nodes
        networklist = list(self.G.edges(data=True))
        networklist.sort()
        input = bytearray(str(networklist), 'utf-8') + bytearray(str(self.excluded), 'utf-8')
        m = hashlib.sha256(input)
        return m.hexdigest()[:8]

    def __update_ginis(self):
        ginis = dict()
        nbcs = dict()
        for u in self.G:
            channel_balance_coeffs = []
            total_balance = 0
            total_capacity = 0
            for v in self.G[u]:
                balance = self.G[u][v]["balance"]
                total_balance += balance
                capacity = self.G[u][v]["capacity"]
                total_capacity += capacity
                cbc = balance / capacity
                channel_balance_coeffs.append(cbc)
            # calculate gini
            gini = Network.gini(channel_balance_coeffs)
            ginis[u] = gini
            # calculate node balance coefficient
            nbc = float(total_balance) / total_capacity
            nbcs[u] = nbc
        nx.set_node_attributes(self.G, ginis, 'gini')
        nx.set_node_attributes(self.G, nbcs, 'nbc')

    def __repr__(self):
        return nx.info(self.G) + '\nMore info?'

    def __str__(self):
        return '<Network with {} nodes and {} channels>'.format(len(self.G), len(self.G.edges))

    def play_rebaloperation(self, op):
        # check if valid rebal op

        self.__history.append(op)

    def rollback_rebaloperation(self):
        assert len(self.__history) > 0, 'Cannot rollback, history is empty'
        op = self.__history.pop()
        logger.info('pooped {}'.format(op))

    @property
    def ops(self):
        return len(self.__history)

    def compute_rebalance_directions(self):
        self.flow = nx.DiGraph()
        for u, v in self.G.edges():
            nbc = self.G.nodes[u]['nbc']
            balance = self.G[u][v]["balance"]
            capacity = self.G[u][v]["capacity"]
            cbc = balance / capacity
            if cbc > nbc:
                amt = int(capacity*(cbc - nbc))
                # print(amt)
                self.flow.add_edge(u, v, liquidity=amt)
        # if both parties want to move an amount over the same channel, remove completely
        delete_edges = []
        for u,v in self.flow.edges():
            if (v,u) in self.flow.edges():
                delete_edges.append((u,v))
                logger.error('channel wants to be rebalanced by both {} and {}'.format(u,v))
        self.flow.remove_edges_from(delete_edges)

    def create_snapshot(self):
        # should be able to store and restore from any intermediate network state
        w = open(self.fingerprint + "_lightning_network", "w")
        for e in self.G.edges(data=True):
            w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                e[0], e[1], e[2]["capacity"], e[2]["balance"],
                e[2]["base"], e[2]["rate"]))
            w.flush()
        w.close()

    @classmethod
    def restore_snapshot(cls, network_file):
        # should be able to store and restore from any intermediate network state
        G = nx.DiGraph()
        f = open(network_file, "r")
        for line in f:
            fields = line[:-1].split("\t")
            if len(fields) == 6:
                s, d, c, a, base, rate = fields
                G.add_edge(s, d, capacity=int(c), balance=int(a), base=float(base), rate=int(rate))
        return cls(G)

    @classmethod
    def parse_clightning(cls, channel_file, init_balance_mode = 'opened'):
        assert init_balance_mode in ['opened', 'normal'], 'Invalid initial balance mode (init_balance_mode)'
        f = open(channel_file, "r")
        logger.debug('parse c-lightning channel dump')
        list_channels = json.load(f)
        raw_channels = list_channels['channels']
        logger.info('{} channels found in file'.format(len(raw_channels)))
        report_raw = len(raw_channels)
        channels = []
        nodes = set()
        edges = []
        simple_edges = set()
        cap_attr,base_attr,rate_attr = dict(), dict(), dict()

        # count occurrences of channel-id and only keep channels that appear twice
        id_occurrence = {}
        for channel in raw_channels:
            sci = channel["short_channel_id"]
            if sci in id_occurrence:
                id_occurrence[sci] += 1
            else:
                id_occurrence[sci] = 1

        report_non_dual = 0
        dupl_channel = set()
        for channel in raw_channels:
            if id_occurrence[channel['short_channel_id']] == 2:
                s = channel['source']
                d = channel['destination']
                if tuple([s,d]) in dupl_channel:
                    # remove channels between the same nodes
                    continue
                else:
                    # add them
                    dupl_channel.add(tuple([s,d]))
                    channels.append(channel)
                    # store some extra channel data for later use
                    cap_attr[(s, d)] = channel['satoshis']
                    base_attr[(s, d)] = channel["base_fee_millisatoshi"] / 1000
                    rate_attr[(s, d)] = channel["fee_per_millionth"]
            else:
                assert id_occurrence[channel['short_channel_id']] == 1, 'other id occurrence than 1 or 2 is not expected'
                report_non_dual += 1
        logger.info('There were {} channels which did not point in both directions. ({} left)'.format(report_non_dual, len(channels)))
        logger.info('init_balance_mode "{}" was chosen'.format(init_balance_mode))
        if init_balance_mode == 'opened':
            # reduce to uni-directional graph
            reduced = set()
            for channel in channels:
                s = channel['source']
                d = channel['destination']
                if s > d:
                    s, d = d, s
                reduced.add(tuple([s, d]))
            logger.info('Reduced to uni-directional graph ({} channels)'.format(len(reduced)))

            # shuffle source<->destination randomly
            shuffled = set()
            for channel in reduced:
                input = bytearray(channel[0] + channel[1], 'utf-8')
                hash_object = hashlib.sha256(input)
                if hash_object.digest()[0] % 2 == 0:
                    shuffled.add(tuple([channel[1], channel[0]]))
                else:
                    shuffled.add(channel)
            logger.info('Shuffled source<>destination ({} channels)'.format(len(reduced)))

            # get max(strongly_connected_component)
            T = nx.DiGraph()
            T.add_edges_from(shuffled)
            strong_conn = max(nx.strongly_connected_components(T), key=len)
            T = T.subgraph(strong_conn).copy()
            logger.info('Found max strongly connected component of length {}'.format(len(list(T.edges))))
            assert strong_conn == max(nx.strongly_connected_components(T), key=len), 'T should now be the strongly connected graph'
            nx.set_edge_attributes(T, cap_attr, 'balance')

            # Reverse the whole graph to replicate channels, set balance to zero
            R = T.reverse(copy=True)
            nx.set_edge_attributes(R, 0, 'balance')
            # Merge the two graphs
            G = nx.compose(T, R)
            logger.info('Replicate reverse edges and merged the network ({} channels)'.format(len(list(G.edges))))

            # set edge properties to new graph
            nx.set_edge_attributes(G, cap_attr, 'capacity')
            nx.set_edge_attributes(G, base_attr, 'base')
            nx.set_edge_attributes(G, rate_attr, 'rate')
            logger.info('Set all the edge properties ({} channels)'.format(len(list(G.edges))))
        elif init_balance_mode == 'normal':
            raise NotImplementedError('mode <normal> is not yet implemented')

        return cls(G)

    @staticmethod
    def gini(x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        # print(x)
        relative_absolute_mean = mean_absolute_differences/np.mean(x)
        # print(relative_absolute_mean)
        return 0.5 * relative_absolute_mean
