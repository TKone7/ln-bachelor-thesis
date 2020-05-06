import random
import json, logging, random
import networkx as nx

FORMAT ='%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format = FORMAT)
logger = logging.getLogger(__name__)
class Network:
    def __init__(self, G):
        self.G = G

    def __repr__(self):
        return nx.info(self.G) + '\nMore info?'

    def __str__(self):
        return '<Network with {} nodes and {} channels>'.format(len(self.G), len(self.G.edges))

    def create_snapshot(self):
        # should be able to store and restore from any intermediate network state
        w = open("lightning_network", "w")
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
        for channel in raw_channels:
            if id_occurrence[channel['short_channel_id']] == 2:
                channels.append(channel)
                # store some extra channel data for later use
                s = channel['source']
                d = channel['destination']
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
                r = random.randint(0, 1)
                if r > 0:
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
