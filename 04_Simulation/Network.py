import hashlib
import json
import logging
import random
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

BASE_FILE = 'lightning_network'
CYCLES_FILE = BASE_FILE + '_cycles'
STATS_GINIS = BASE_FILE + '_stats_ginis'
STATS = BASE_FILE + '_stats'

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


class Network:
    def __init__(self, G, participation, cycles4=None, cycles5=None, iteration=1, selection='random'):
        self.G = G
        self.flow = None
        self.participation = participation
        self.iteration = iteration
        self.selection = selection
        self.__history = []
        self.__history_gini = []
        self.__stats = {}
        self.__all_pair_shortest_paths = []
        self.__all_pair_max_flow = dict()

        # calculate initial gini coefficients for all nodes
        self.__update_ginis()
        # calculate shortest paths (will not change in future)
        self.__compute_all_pair_shortest_paths()
        self.__compute_all_pair_max_flow()
        # exclude certain nodes
        if selection == 'random':
            # random1:
            # random.seed(100)
            # random2:
            #random.seed(200)
            random.seed(iteration * 100)
            nodes_sorted = list(self.G.nodes)
            nodes_sorted.sort()
            excl = random.sample(nodes_sorted, int(len(nodes_sorted) * (1 - participation)))
        else:
            raise NotImplementedError('Only random selection of participants is implemented. But \'{}\' was tried.'.format(selection))

        self.__excluded = set(excl)

        # reduce the cycles if available
        if participation != 1 and cycles4:
            # reduce the cycles available
            cycles4 = [c for c in cycles4 if not (set(c) & self.__excluded)]
        self.__cycles4 = cycles4 if cycles4 else []
        self.__cycles5 = cycles5 if cycles5 else []

        # calculate rebalance network
        self.__compute_rebalance_network()

        self.fingerprint = self.__fingerprint()
        self.experiment_name = self.fingerprint + '_' + str(int(participation * 100)) + '_' + selection + '_' + str(iteration)
        logger.info('Fingerprint is {}'.format(self.fingerprint))

    def __fingerprint(self):
        def s(tup):
            return tup[0] + tup[1] + str(tup[2]) + str(tup[3]) + str(tup[4]) + str(tup[5])

        # calculate a fingerprint for the initial network state. Should include:
        # - nodes / edges / attributes
        # - excluded nodes
        networklist = list(self.G.edges(data=True))
        network = [(n[0], n[1], n[2]['capacity'], n[2]['balance'], n[2]['base'], n[2]['rate']) for n in networklist]
        network.sort(key=s)
        input = bytearray(str(network), 'utf-8')  # + bytearray(str(self.__excluded), 'utf-8')
        m = hashlib.sha256(input)
        return m.hexdigest()[:8]

    def __update_ginis(self):
        for u in self.G:
            # calculate gini
            self.__update_node_gini(u)

    def __update_node_gini(self, node):
        channel_balance_coeffs = []
        weights = []
        node_balance = 0
        node_capacity = 0
        for v in self.G[node]:
            balance = self.G[node][v]["balance"]
            node_balance += balance
            capacity = self.G[node][v]["capacity"]
            node_capacity += capacity
            cbc = float(balance) / capacity
            channel_balance_coeffs.append(cbc)
            weights.append(capacity)
        # calculate gini
        gini = Network.gini(channel_balance_coeffs)
        # gini = Network.weighted_gini(channel_balance_coeffs,weights)
        # calculate node balance coefficient
        nbc = float(node_balance) / node_capacity
        self.G.nodes[node]['gini'] = gini
        assert ('nbc' not in self.G.nodes[node]) or self.G.nodes[node][
            'nbc'] == nbc, 'node balance coefficients should never change'
        self.G.nodes[node]['nbc'] = nbc

    def __rearrange(self, init, circle):
        raise ValueError('rearrange id deprecated and can no longer be used')
        # init_idx = circle.index(init)
        # new_circle = []
        # for i in range(len(circle)):
        #     curr = (i + init_idx) % len(circle)
        #     new_circle.append(circle[curr])
        # new_circle.append(init)
        # return new_circle

    def __update_channel(self, op, rev=False):
        # takes care of the channel balances
        amount = op[0] if not rev else op[0] * -1
        circle = op[1]
        for i in range(len(circle) - 1):
            src = circle[i]
            dest = circle[i + 1]
            self.G[src][dest]['balance'] -= amount
            self.G[dest][src]['balance'] += amount
            assert np.sign(
                self.flow[src][dest]['liquidity']) == 1 or rev, 'Liquidities should constantly go towards zero.'
            assert np.sign(
                self.flow[dest][src]['liquidity']) == -1 or rev, 'Liquidities should constantly go towards zero.'
            self.flow[src][dest]['liquidity'] -= amount
            self.flow[dest][src]['liquidity'] += amount
        [self.__update_node_gini(n) for n in circle[:-1]]

    def __repr__(self):
        return nx.info(self.G) + '\nMore info?'

    def __str__(self):
        return '<Network with {} nodes and {} channels>'.format(len(self.G), len(self.G.edges))

    def play_rebaloperation(self, op):
        # check if valid rebal op
        if isinstance(op[0], int) and isinstance(op[1], list):
            self.__update_channel(op)
            self.__history.append(op)

            mean_gini = self.mean_gini
            self.__history_gini.append(mean_gini)

            # store network statistics (success measures) for every 0.01 gini improvement
            stats_key = "{0:.2f}".format(mean_gini)
            if stats_key not in self.__stats:
                self.__stats[stats_key] = self.calculate_routing_stats()
        else:
            logger.error('this is not a valid opertaion to perform: {}'.format(op))

    def rollback_rebaloperation(self, nr=1):
        raise ValueError('rollback id deprecated and can no longer be used')
        # for i in range(nr):
        # assert len(self.__history) > 0, 'Cannot rollback, history is empty'
        # op = self.__history.pop()
        # self.__update_channel(op, rev=True)
        # # logger.info('pooped {}'.format(op))

    @property
    def ops(self):
        raise ValueError('ops is deprecated and can no longer be used')
        # return len(self.__history)

    @property
    def mean_gini(self):
        ginis_dict = nx.get_node_attributes(self.G, 'gini')
        ginis = list(ginis_dict.values())
        return np.mean(ginis)

    def calculate_routing_stats(self):
        # (median_payment_amount, success_rate)
        stats = {}
        amounts = []
        for path in self.__all_pair_shortest_paths:
            liqs = []
            for i in range(len(path) - 1):  # exclude last
                src = path[i]
                dest = path[i + 1]
                liqs.append(self.G[src][dest]['balance'])
            amount = int(min(liqs))
            amounts.append(amount)
        # calc stats
        median_payment_amount = np.median(amounts)
        zero_amounts = amounts.count(0)
        success_rate = (len(amounts) - zero_amounts) / len(amounts)
        stats['median_payment_amount'] = median_payment_amount
        stats['success_rate'] = success_rate
        return stats

    def __compute_all_pair_shortest_paths(self):
        apsp = nx.all_pairs_dijkstra_path(self.G, weight='base', cutoff=20)
        for paths in apsp:
            for k, v in paths[1].items():
                if len(v) > 1:  # ignore paths with only one element (is source element)
                    self.__all_pair_shortest_paths.append(v)
    def __compute_all_pair_max_flow(self):
        apmf = dict()
        nodes = len(self.G.nodes)
        conns = (nodes*(nodes-1))/2
        done = 0
        logger.info('todo: {}'.format(nodes))
        for u in self.G.nodes:
            s = dict()
            for v in self.G.nodes:
                if u != v:
                    if v in apmf.keys():
                        s[v] = apmf[v][u]
                    else:
                        s[v], _ = nx.maximum_flow(self.G, u, v)
                        done +=1
            apmf[u] = s
            logger.info('status {}%'.format((done/conns)*100))
        self.__all_pair_max_flow = apmf

    def __compute_rebalance_network(self):
        # This calculates a new graph of desired flows by each node.
        # Only for nodes which participate
        self.flow = nx.DiGraph()
        delete_edges = []
        for u, v in self.G.edges():
            if u in self.__excluded or v in self.__excluded:
                continue
            nbc = self.G.nodes[u]['nbc']
            balance = self.G[u][v]["balance"]
            capacity = self.G[u][v]["capacity"]
            cbc = balance / capacity
            # amount can be above or below zero (below signals desire for receiving balance)
            amt = int(capacity * (cbc - nbc))
            if (v, u) in self.flow.edges():
                amt_cp = self.flow[v][u]['liquidity']
                if np.sign(amt) == np.sign(amt_cp):
                    logger.error('signs of desired channel flows are equal between {} and {}'.format(u, v))
                    delete_edges.append((v, u))
                    continue
                common = min(abs(amt), abs(amt_cp))
                amt_cp = common * np.sign(amt_cp)
                amt = common * np.sign(amt)
                self.flow[v][u]['liquidity'] = amt_cp
            # if cbc > nbc:
            # check if reverse channel is already in list, if yes, remove both

            self.flow.add_edge(u, v, liquidity=amt)

        if len(delete_edges) > 0:
            logger.debug('will delete {} edges'.format(len(delete_edges)))
        self.flow.remove_edges_from(delete_edges)

    def compute_circles(self, force=False):
        # remove edges with liquidity 0 since they don't want to route anything anymore
        # TODO, can probably be removed since we only compute these once in the beginning. Not after rebalancings took place
        # zero_flow = [e for e in self.flow.edges(data=True) if e[2]['liquidity'] == 0]
        # self.flow.remove_edges_from(zero_flow)
        if self.__cycles4 != [] and not force:
            logger.info('There are already cycles. Abort recomputing them.')
            return
        # All possible circles to conduct rebalancing
        # Need to be calculated only once (per network) since there never will be more / others
        cycles4 = []
        # seen = set()
        # cycles5 = []
        pos_edges = [e for e in self.flow.edges(data=True) if e[2]['liquidity'] > 0]
        pos_flow = nx.DiGraph()
        pos_flow.add_edges_from(pos_edges)
        for i, (u, v) in enumerate(pos_flow.edges):
            paths = [p for p in nx.all_simple_paths(pos_flow, v, u, 3)]
            [cycles4.append(p) for p in paths if len(p) <= 4]
            # for p in paths:
            # if p not in cycles4:
            #     ind = p.index(min(p))
            #     po = [p[(ind + e) % len(p)] for e in range(len(p))] # set the same node as start of the cycle to avoid duplicate cycles
            #     if po not in cycles4:
            #         cycles4.append(po)
            if i % 100 == 0:
                logger.info('{} edges checked for circles'.format(i))

        logger.debug('There are {} circles of length 4 or less'.format(len(cycles4)))
        # logger.debug('There are {} circles of length 5 or less'.format(len(cycles4)))
        self.__cycles4 = cycles4
        # self.__cycles5 = cycles4
        random.seed(10)
        self.__cycles4.sort()
        random.shuffle(self.__cycles4)
        # store the results to a file
        self.__store_cycles()

    def rebalance(self, max_ops=100000, amount_coeff=1):
        nr_executed = 0
        executed_this_time = 0
        if len(self.__cycles4) <= 0:
            logger.error('There are no ops to rebalance. Please compute operations first')
            return
        before = len(self.__history)

        while nr_executed < max_ops:
            local_before = len(self.__history)
            for circle in self.__cycles4:
                if nr_executed >= max_ops:
                    break
                # check what is currently the max amount (maybe divide)
                liqs = [self.flow[e][circle[(i + 1) % len(circle)]]['liquidity'] for i, e in enumerate(circle)]
                amount = int(min(liqs) * amount_coeff)
                if amount < 1:
                    continue
                new_circle = circle.copy()
                new_circle.insert(0, circle[-1])
                # logger.info('Rebalance {} sat over circle {}'.format(amount, ", ".join(new_circle)))

                self.play_rebaloperation((amount, new_circle))
                nr_executed += 1
                if nr_executed % 1000 == 0:
                    logger.info('{} operations: Networks mean-gini is {}'.format(nr_executed, self.__history_gini[-1]))
            if len(self.__history) - local_before == 0:
                break

        after = len(self.__history)
        logger.info('Successfuly rebalanced {} operations'.format(after - before))

    def __store_cycles(self):
        # check for operations to store
        if len(self.__cycles4) > 0:
            if not os.path.isdir(self.fingerprint):
                logger.error('Folder {} is not available. Create snapshot first.'.format(self.fingerprint))
            cycles_file = os.path.join(self.fingerprint, self.fingerprint + '_' + CYCLES_FILE + '_4' + '_' + str(
                int(self.participation * 100)))
            f = open(cycles_file, 'w')
            for circle in self.__cycles4:
                f.write(" ".join(circle) + '\n')
                f.flush()
            f.close()

    def create_snapshot(self):
        # create folder for each specific network
        if not os.path.isdir(self.fingerprint):
            os.makedirs(self.fingerprint)
        network_file = os.path.join(self.fingerprint, self.fingerprint + '_' + BASE_FILE)
        w = open(network_file, "w")
        for e in self.G.edges(data=True):
            w.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                e[0], e[1], e[2]["capacity"], e[2]["balance"],
                e[2]["base"], e[2]["rate"]))
            w.flush()
        w.close()

    def store_experiment_result(self):
        # iteration=1, selection='random'
        assert self.__stats and self.__history_gini, 'Statistics are empty, no experiment was performed. Cannot store results.'
        assert self.__history, 'No experiment was performed. Cannot store results.'
        assert self.flow and self.G, 'The network is empty. Cannot store results.'
        if not os.path.isdir(self.fingerprint):
            logger.error('Folder {} is not available. Create snapshot first.'.format(self.fingerprint))
        stats_file = os.path.join(self.fingerprint, STATS + '_' + self.experiment_name) + '.json'
        ginis_file = os.path.join(self.fingerprint, STATS_GINIS + '_' + self.experiment_name) + '.json'
        graph_file = os.path.join(self.fingerprint, 'NETWORK' + '_' + self.experiment_name)
        flow_graph_file = os.path.join(self.fingerprint, 'FLOW' + '_' + self.experiment_name)

        with open(stats_file, "w") as f:
            json.dump(self.stats(), f)
        with open(ginis_file, "w") as f:
            json.dump(self.gini_hist_data(), f)
        nx.write_gpickle(self.G, graph_file)
        nx.write_gpickle(self.flow, flow_graph_file)

    # get data for plotting
    def gini_distr_data(self):
        ginis_dict = nx.get_node_attributes(self.G, 'gini')
        return list(ginis_dict.values())

    def gini_hist_data(self):
        return self.__history_gini

    def stats(self):
        return self.__stats

    def apmf(self):
        return self.__all_pair_max_flow

    def plot_gini_distr_hist(self, filename='gini_distr_hist'):
        # plot gini distribution
        ginis = self.gini_distr_data()
        plt.hist(ginis, bins=20)
        plt.title("Distribution of nodes Ginicoefficients")
        plt.xlabel("Imbalance (Ginicoefficients $G_v$) of nodes")
        plt.ylabel("Frequency $C(G_v)$")
        plt.grid()
        Network.__store_chart(self.fingerprint, filename + '_' + self.experiment_name)

    def plot_gini_vs_rebalops(self, filename='gini_vs_rebalops'):
        # plot gini over time
        hist = self.gini_hist_data()
        plt.plot(hist, label='100% of min amount', linewidth=3)
        plt.title("Network imbalance over time (successfull rebalancing operations)")
        plt.xlabel("Number of successfull rebalancing operations (logarithmic)")
        plt.ylabel("Network imbalance (G)")
        plt.xscale("log")
        plt.xlim(100, 10000000)
        plt.grid()
        plt.legend(loc="upper right")
        Network.__store_chart(self.fingerprint, filename + '_' + self.experiment_name)

    def plot_paymentsize_vs_imbalance(self, filename='median_payment_size'):
        # plot stats per 0.01 gini
        stats = self.stats()
        [s['median_payment_amount'] for s in stats.values()]
        plt.plot(list(stats.keys())[::-1], [s['median_payment_amount'] for s in stats.values()][::-1],
                 label='cycles of length 4', linewidth=3)
        plt.title("Comparing Network imbalance with possible payment size")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Median possible payment size [satoshi]")
        plt.legend(loc="upper right")
        plt.grid()
        Network.__store_chart(self.fingerprint, filename + '_' + self.experiment_name)

    def plot_successrate_vs_imbalance(self, filename='successrate_vs_imbalance'):
        stats = self.stats()
        plt.plot(list(stats.keys())[::-1], [s['success_rate'] for s in stats.values()][::-1],
                 label='cycles of length 4', linewidth=3)
        plt.title("Comparing Network imbalance with success rate of random payments")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Success rate of random payment")
        plt.grid()
        plt.legend(loc="lower left")
        Network.__store_chart(self.fingerprint, filename + '_' + self.experiment_name)

    @classmethod
    def restore_snapshot(cls, fingerprint, participation=1, is_file=False, iteration=1, selection='random'):
        # should be able to store and restore from any intermediate network state
        G = nx.DiGraph()
        cycles4 = None
        cycles5 = None
        if is_file:
            f = open(fingerprint, "r")
        else:
            network_file = os.path.join(fingerprint, fingerprint + "_" + BASE_FILE)
            f = open(network_file, "r")
            # check also for a file with operations CYCLES_FILE
            try:
                cycles_file = os.path.join(fingerprint, fingerprint + "_" + CYCLES_FILE + '_4')
                c = open(cycles_file, "r")
                cycles4 = []
                for line in c:
                    cycle = line.replace('\n', '').split((' '))
                    cycles4.append(cycle)
            except:
                logger.info('No file with precomputed cycles - length 4 - found. ')
        for line in f:
            fields = line[:-1].split("\t")
            if len(fields) == 6:
                s, d, c, a, base, rate = fields
                G.add_edge(s, d, capacity=int(c), balance=int(a), base=int(base), rate=int(rate))
        N = cls(G, participation, cycles4, cycles5, iteration=iteration, selection=selection)
        assert fingerprint == N.fingerprint or is_file, 'Fingerprints of stored and restored network are not equal'
        return N

    @classmethod
    def restore_result(cls, fingerprint, participation, iteration=1, selection='random'):
        if not os.path.isdir(fingerprint):
            logger.error('Folder {} is not available. Create snapshot first.'.format(fingerprint))
        part100 = str(int(participation * 100))
        experiment_name = fingerprint + '_' + str(int(participation * 100)) + '_' + selection + '_' + str(iteration)
        #        self.experiment_name = self.fingerprint + '_' + str(int(participation * 100)) + selection + '_' + iteration
        stats_file = os.path.join(fingerprint, STATS + '_' + experiment_name) + '.json'
        ginis_file = os.path.join(fingerprint, STATS_GINIS + '_' + experiment_name) + '.json'
        graph_file = os.path.join(fingerprint, 'NETWORK' + '_' + experiment_name)
        flow_graph_file = os.path.join(fingerprint, 'FLOW' + '_' + experiment_name)

        G = nx.read_gpickle(graph_file)
        N = cls(G, participation, iteration=iteration, selection=selection)
        N.fingerprint = fingerprint
        N.flow = nx.read_gpickle(flow_graph_file)
        with open(stats_file, "r") as f:
            stats = json.load(f)
            N.__stats = stats
        with open(ginis_file, "r") as f:
            ginis = json.load(f)
            N.__history_gini = ginis
        return N

    @classmethod
    def parse_clightning(cls, channel_file, participation=1, init_balance_mode='opened'):
        assert init_balance_mode in ['opened', 'normal'], 'Invalid initial balance mode (init_balance_mode)'
        f = open(channel_file, "r")
        logger.debug('parse c-lightning channel dump')
        list_channels = json.load(f)
        raw_channels = list_channels['channels']
        logger.info('{} channels found in file'.format(len(raw_channels)))
        channels = []
        cap_attr, base_attr, rate_attr = dict(), dict(), dict()

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
                if tuple([s, d]) in dupl_channel:
                    # remove channels between the same nodes
                    continue
                else:
                    # add them
                    dupl_channel.add(tuple([s, d]))
                    channels.append(channel)
                    # store some extra channel data for later use
                    cap_attr[(s, d)] = int(channel['satoshis'])
                    base_attr[(s, d)] = int(channel["base_fee_millisatoshi"])
                    rate_attr[(s, d)] = int(channel["fee_per_millionth"])
            else:
                assert id_occurrence[
                           channel['short_channel_id']] == 1, 'other id occurrence than 1 or 2 is not expected'
                report_non_dual += 1
        logger.info('There were {} channels which did not point in both directions. ({} left)'.format(report_non_dual,
                                                                                                      len(channels)))
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
            assert strong_conn == max(nx.strongly_connected_components(T),
                                      key=len), 'T should now be the strongly connected graph'
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

        return cls(G, participation=participation)

    @staticmethod
    def gini(x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        # print(x)
        relative_absolute_mean = mean_absolute_differences / np.mean(x)
        # print(relative_absolute_mean)
        return 0.5 * relative_absolute_mean

    @staticmethod
    def weighted_gini(x, weights=None):
        if weights is None:
            weights = np.ones_like(x)
        count = np.multiply.outer(weights, weights)
        mad = np.abs(np.subtract.outer(x, x) * count).sum() / count.sum()
        rmad = mad / np.average(x, weights=weights)
        return 0.5 * rmad

    @staticmethod
    def plot_gini_vs_rebalops_merge(networks, filename='gini_vs_rebalops_merge'):
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            hist = n.gini_hist_data()
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            plt.plot(hist, label=lbl, linewidth=3)
        plt.title("Network imbalance over time (successfull rebalancing operations)")
        plt.xlabel("Number of successfull rebalancing operations (logarithmic)")
        plt.ylabel("Network imbalance (G)")
        plt.xscale("log")
        plt.xlim(100, 10000000)
        plt.grid()
        plt.legend(loc="upper right")
        Network.__store_chart(fingerprint, filename)

    @staticmethod
    def plot_paymentsize_vs_imbalance_merge(networks, filename='median_payment_size_merge'):
        # plot stats per 0.01 gini
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['median_payment_amount'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
            #plt.plot(list(stats.keys())[::-1], [s['median_payment_amount_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
        plt.title("Comparing Network imbalance with possible payment size")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Median possible payment size [satoshi]")
        plt.legend(loc="upper right")
        plt.grid()
        Network.__store_chart(fingerprint, filename)

    @staticmethod
    def plot_successrate_vs_imbalance_merge(networks, filename='successrate_vs_imbalance_merge'):
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['success_rate'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
            #plt.plot(list(stats.keys())[::-1], [s['success_rate_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
        plt.title("Comparing Network imbalance with success rate of random payments")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Success rate of random payment")
        plt.grid()
        plt.legend(loc="lower left")
        Network.__store_chart(fingerprint, filename)

    @staticmethod
    def plot_gini_vs_participation(networks, filename='gini_vs_participation'):
        def byParticipation(n):
            return n.participation
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        networks.sort(key=byParticipation)
        data = [n.gini_distr_data() for n in networks]
        lbl = [str(n.participation*100)+'%' for n in networks]
        # Create a figure instance

        plt.boxplot(data, labels=lbl)
        Network.__store_chart(fingerprint, filename)

    @classmethod
    def condense_networks(cls, networks):
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot condense different networks together'
        assert len(set([n.participation for n in networks])) == 1, 'You cannot condense different participations together'
        assert len(set([n.selection for n in networks])) == 1, 'You cannot condense different selection together'
        fingerprint = networks[0].fingerprint
        participation = networks[0].participation
        selection = networks[0].selection
        N = cls(networks[0].G, participation, iteration=0, selection=selection)
        N.fingerprint = fingerprint

        keys = set()
        figures = set()
        [keys.update(n.stats().keys()) for n in networks]
        for n in networks:
            for v in list(n.stats().values()):
                figures.update(v.keys())
        print('keys {}, figures {}'.format(keys, figures))
        key_l = list(keys)
        key_l.sort(reverse=True)

        avg_stats = dict()
        for k in key_l:
            avg_figures = dict()
            for f in figures:
                available = [n for n in networks ]
                avg_figures[f] = np.average([n.stats()[k][f] for n in networks if k in n.stats()])
                avg_figures[f+'_std'] = np.std([n.stats()[k][f] for n in networks if k in n.stats()])
            avg_stats[k] = avg_figures
        N.__stats = avg_stats

        # gini history
        entries = max([len(n.gini_hist_data()) for n in networks])
        avg_gini_hist = [np.average([n.gini_hist_data()[e] for n in networks if len(n.gini_hist_data()) >= e+1]) for e in range(entries)]
        N.__history_gini = avg_gini_hist
        return N

    @staticmethod
    def __store_chart(fingerprint, filename, types=('pdf', 'png')):
        charts = 'charts'
        path = os.path.join(fingerprint, charts)
        if not os.path.isdir(path):
            os.makedirs(path)
        for file_type in types:
            chart_file = os.path.join(path, fingerprint + '_' + filename + '.' + file_type)
            plt.savefig(chart_file)
        plt.close()
