import hashlib
import json
import logging
import random
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

BASE_FILE = 'lightning_network'
CYCLES_FILE = BASE_FILE + '_cycles'
SIMPLE_PATH = BASE_FILE + '_simple_path'
STATS_GINIS = BASE_FILE + '_stats_ginis'
FINAL_STATS = BASE_FILE + '_final_stats'
STATS = BASE_FILE + '_stats'

MICRO_PAYMENT_SIZE = 10000
NORMAL_PAYMENT_SIZE = 100000

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


class Network:
    def __init__(self, G, cycles4=None):
        self.G = G
        self.flow = None
        self.participation = None
        self.experiment_name = None
        self.__history = []
        self.__history_gini = []
        self.__stats = {}
        self.__all_pair_max_flow = dict()
        self.__excluded = []
        self.__cycles4 = cycles4 if cycles4 else []
        self.__all_pair_shortest_paths = None
        # self.__all_pair_simple_paths = None
        # self.__k_shortest_path = None
        self.incl_capacity = None
        self.total_capacity = None
        self.nr_nodes = None
        self.nr_participating_nodes = None
        self.end_mean_gini = None
        self.end_std_mean_gini = None
        self.fingerprint = self.__fingerprint()
        logger.info('Fingerprint is {}'.format(self.fingerprint))

    def __repr__(self):
        return nx.info(self.G)

    def __str__(self):
        return '<Network with {} nodes and {} channels>'.format(len(self.G), len(self.G.edges))

    # getter / setter
    def set_experiment_name(self, name):
        self.experiment_name = self.fingerprint + '_' + name

    def set_participation(self, participation):
        self.participation = participation

    def gini_distr_data(self):
        ginis_dict = nx.get_node_attributes(self.G, 'gini')
        return list(ginis_dict.values())

    def gini_hist_data(self):
        return self.__history_gini

    def stats(self):
        return self.__stats

    def set_stats(self, stats):
        self.__stats = stats

    def history_gini(self):
        return self.__history_gini

    def set_history_gini(self, history_gini):
        self.__history_gini = history_gini

    def history(self):
        return self.__history

    def apmf(self):
        return self.__all_pair_max_flow

    # Private mthods
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

    def __update_all_ginis(self):
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

    def __update_channel(self, tx, rev=False):
        # takes care of the channel balances
        amount = tx[0] if not rev else tx[0] * -1
        circle = tx[1]
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

    def __compute_all_pair_shortest_paths(self):
        all_pair_shortest_paths = []
        apsp = nx.all_pairs_dijkstra_path(self.G, weight='base')
        for paths in apsp:
            for k, v in paths[1].items():
                if len(v) > 1:  # ignore paths with only one element (is source element)
                    all_pair_shortest_paths.append(v)
        return all_pair_shortest_paths

    # def __compute_k_shortest_path(self):
    #     self.__k_shortest_path = 'bb'
    #     C = self.G.copy()
    #     cnt = 0
    #     for nodepair in self.__all_pair_shortest_paths:
    #         src = nodepair[0]
    #         dest = nodepair[-1]
    #         ksp = algo.ksp_yen(C, src, dest, 10, 'base')
    #         cnt += 1
    #         if max([p['cost'] for p in ksp]) > 2000:
    #             logger.info('high cost detected between {} and {}'.format(src, dest))
    #         if cnt % 1000 == 0:
    #             logger.info('k-shortest path status: {}%'.format(100 / len(self.__all_pair_shortest_paths) * cnt))

    def __compute_all_pair_simple_paths(self):
        assert self.__all_pair_shortest_paths, 'Shortest paths between all pair must be calculated first'
        try:  # to load from file first
            simple_path_file = os.path.join(self.fingerprint, self.fingerprint + '_' + SIMPLE_PATH) + '.json'
            with open(simple_path_file, "r") as f:
                sp = json.load(f)
                return sp
        except OSError as e:
            simplepath = dict()
            cnt = 0

            for nodepair in self.__all_pair_shortest_paths:
                src = nodepair[0]
                dest = nodepair[-1]
                k_shortest_path = list(islice(nx.all_simple_paths(self.G, src, dest, 6), 10))
                cnt += 1
                simplepath[cnt] = k_shortest_path
                if cnt % 1000 == 0:
                    logger.info('retry number status: {}%'.format(100/len(self.__all_pair_shortest_paths)*cnt))
            # store result to file
            simple_path_file = os.path.join(self.fingerprint, self.fingerprint + '_' + SIMPLE_PATH) + '.json'
            with open(simple_path_file, "w") as f:
                json.dump(simplepath, f)
            return simplepath

    def __compute_multi_payment_stat(self):
        raise DeprecationWarning
        total_success_cnt = []
        total_micro_cnt = []
        total_normal_cnt = []

        for _, path_list in self.__all_pair_simple_paths.items():
            success_cnt = 0
            micro_cnt = 0
            normal_cnt = 0
            for path in path_list:
                max_flow = self.__max_flow_along_path(path)
                if max_flow > 0:
                    success_cnt += 1
                if max_flow > MICRO_PAYMENT_SIZE:
                    micro_cnt += 1
                if max_flow > NORMAL_PAYMENT_SIZE:
                    normal_cnt += 1
            total_success_cnt.append(success_cnt)
            total_micro_cnt.append(micro_cnt)
            total_normal_cnt.append(normal_cnt)
        return total_success_cnt, total_micro_cnt, total_normal_cnt

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

    def __max_flow_along_path(self, path):
        assert len(path) > 1, 'Path must include at least two nodes, to calc the max flow.'
        liqs = []
        for i in range(len(path) - 1):  # exclude last
            src = path[i]
            dest = path[i + 1]
            liqs.append(self.G[src][dest]['balance'])
        return int(min(liqs))

    def __store_cycles(self):
        # check for operations to store
        if len(self.__cycles4) > 0:
            if not os.path.isdir(self.fingerprint):
                logger.error('Folder {} is not available. Create snapshot first.'.format(self.fingerprint))
            cycles_file = os.path.join(self.fingerprint, self.fingerprint + '_' + CYCLES_FILE + '_4')
            f = open(cycles_file, 'w')
            for circle in self.__cycles4:
                f.write(" ".join(circle) + '\n')
                f.flush()
            f.close()
    def __meta_info(self):
        info = {
            'total_capacity': self.total_capacity,
            'available_capacity': self.incl_capacity,
            'total_nodes': len(self.G.nodes),
            'participating_nodes': self.nr_participating_nodes
        }
        return info
    def __set_meta_info(self, meta_info):
        self.total_capacity = meta_info['total_capacity']
        self.incl_capacity = meta_info['available_capacity']
        self.nr_participating_nodes = meta_info['participating_nodes']
        self.nr_nodes = meta_info['total_nodes']
    def __set_final_stats(self, final_stats):
        self.__final_stats = final_stats

    def exclude(self, excl_list):
        assert self.__cycles4, 'Cannot exclude nodes before the cycles are not calculated. Run "compute_circles()" first.'

        self.__excluded = set(excl_list)
        cycles4 = [c for c in self.__cycles4 if not (set(c) & self.__excluded)]
        self.__cycles4 = cycles4
        # calculate the capacity of all included channels
        visited_edge = set()
        incl_capacity = 0
        total_capacity = 0
        for u, v in self.G.edges:
            if (v,u) not in visited_edge:
                visited_edge.add((u, v))
                curr_cap = self.G[u][v]['capacity']
                total_capacity += curr_cap
                if not set([u, v]) & self.__excluded:
                    incl_capacity += curr_cap

        self.incl_capacity = incl_capacity
        self.total_capacity = total_capacity
        self.nr_participating_nodes = len(set(self.G.nodes) - self.__excluded)
        # calculate initial gini coefficients for all nodes
        self.__update_all_ginis()

        # recalculate rebalance network
        self.__compute_rebalance_network()

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

    @property
    def ops(self):
        raise ValueError('ops is deprecated and can no longer be used')

    @property
    def mean_gini(self):
        ginis_dict = nx.get_node_attributes(self.G, 'gini')
        ginis = list(ginis_dict.values())
        return np.mean(ginis)

    def calculate_routing_stats(self):
        # (median_payment_amount, success_rate, median_retry)
        stats = {}
        amounts = []
        # Get list with all amounts along all shortest paths
        for path in self.__all_pair_shortest_paths:
            max_flow = self.__max_flow_along_path(path)
            amounts.append(max_flow)

        # Get successful payments on 20 simple paths between all pairs
        # total_success_cnt, total_micro_cnt, total_normal_cnt = self.__compute_multi_payment_stat()

        # calc stats
        median_payment_amount = np.median(amounts)
        zero_amounts = amounts.count(0)
        success_rate = (len(amounts) - zero_amounts) / len(amounts)
        # median_success = np.median(total_success_cnt)
        # median_micro = np.median(total_micro_cnt)
        # median_normal = np.median(total_normal_cnt)
        # store stats
        stats['median_payment_amount'] = median_payment_amount
        stats['success_rate'] = success_rate
        # stats['median_success'] = median_success
        # stats['median_micro'] = median_micro
        # stats['median_normal'] = median_normal
        return stats

    def compute_circles(self, force=False):
        # remove edges with liquidity 0 since they don't want to route anything anymore
        # TODO, can probably be removed since we only compute these once in the beginning. Not after rebalancings took place
        # zero_flow = [e for e in self.flow.edges(data=True) if e[2]['liquidity'] == 0]
        # self.flow.remove_edges_from(zero_flow)
        if self.__cycles4 != [] and not force:
            logger.info('There are already cycles. Abort recomputing them.')
            return
        # calculate initial gini coefficients for all nodes
        self.__update_all_ginis()
        if not self.flow:
            # in case the flow is not yet calculated do so
            self.__compute_rebalance_network()
        # All possible circles to conduct rebalancing
        # Need to be calculated only once (per network) since there never will be more / others
        cycles4 = []
        # seen = set()
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
        random.seed(10)
        self.__cycles4.sort()
        random.shuffle(self.__cycles4)
        # store the results to a file
        self.__store_cycles()

    def rebalance(self, max_ops=100000, amount_coeff=1):
        # calculate shortest paths first (will change in future)
        self.__all_pair_shortest_paths = self.__compute_all_pair_shortest_paths()
        logger.info('finished shortest path calculation')
        # calculate (if not loaded) simple paths between all node pairs
        # self.__all_pair_simple_paths = self.__compute_all_pair_simple_paths()

        # self.__compute_k_shortest_path()
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
        assert self.__stats and self.__history_gini, 'Statistics are empty, no experiment was performed. Cannot store results.'
        assert self.__history, 'No experiment was performed. Cannot store results.'
        assert self.flow and self.G, 'The network is empty. Cannot store results.'
        # store final results
        self.__final_stats = self.calculate_routing_stats()

        if not os.path.isdir(self.fingerprint):
            logger.error('Folder {} is not available. Create snapshot first.'.format(self.fingerprint))
        final_stats_file = os.path.join(self.fingerprint, FINAL_STATS + '_' + self.experiment_name) + '.json'
        stats_file = os.path.join(self.fingerprint, STATS + '_' + self.experiment_name) + '.json'
        ginis_file = os.path.join(self.fingerprint, STATS_GINIS + '_' + self.experiment_name) + '.json'
        meta_file = os.path.join(self.fingerprint, 'META' + '_' + self.experiment_name) + '.json'
        graph_file = os.path.join(self.fingerprint, 'NETWORK' + '_' + self.experiment_name)
        flow_graph_file = os.path.join(self.fingerprint, 'FLOW' + '_' + self.experiment_name)

        with open(final_stats_file, "w") as f:
            json.dump(self.__final_stats, f)
        with open(stats_file, "w") as f:
            json.dump(self.stats(), f)
        with open(ginis_file, "w") as f:
            json.dump(self.gini_hist_data(), f)
        with open(meta_file, "w") as f:
            meta_info = self.__meta_info()
            json.dump(meta_info, f)
        nx.write_gpickle(self.G, graph_file)
        nx.write_gpickle(self.flow, flow_graph_file)

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

    def plot_payments_vs_imbalance(self, filename='payments_vs_imbalance'):
        raise DeprecationWarning
        stats = self.stats()
        plt.plot(list(stats.keys())[::-1], [s['median_success'] for s in stats.values()][::-1],
                 label='Route min 1 sat', linewidth=3)
        plt.plot(list(stats.keys())[::-1], [s['median_micro'] for s in stats.values()][::-1],
                 label='Route min {} sats'.format(MICRO_PAYMENT_SIZE), linewidth=3)
        plt.plot(list(stats.keys())[::-1], [s['median_normal'] for s in stats.values()][::-1],
                 label='Route min {} sats'.format(NORMAL_PAYMENT_SIZE), linewidth=3)
        plt.title("Comparing Network imbalance with successful payments")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("median nr of payments (out of 10)")
        plt.grid()
        plt.legend(loc="lower left")
        Network.__store_chart(self.fingerprint, filename + '_' + self.experiment_name)

    @classmethod
    def restore_snapshot(cls, fingerprint, is_file=False):
        # should be able to store and restore from any intermediate network state
        G = nx.DiGraph()
        cycles4 = None
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
                    cycle = line.replace('\n', '').split(' ')
                    cycles4.append(cycle)
            except OSError as e:
                logger.info('No file with precomputed cycles - length 4 - found. ')
        for line in f:
            fields = line[:-1].split("\t")
            if len(fields) == 6:
                s, d, c, a, base, rate = fields
                G.add_edge(s, d, capacity=int(c), balance=int(a), base=int(base), rate=int(rate))

        N = cls(G, cycles4)
        assert fingerprint == N.fingerprint or is_file, 'Fingerprints of stored and restored network are not equal'
        return N

    @classmethod
    def restore_result_by_name(cls, fingerprint, experiment_name):
        if not os.path.isdir(fingerprint):
            logger.error('Folder {} is not available. Create snapshot first.'.format(fingerprint))
        #        self.experiment_name = self.fingerprint + '_' + str(int(participation * 100)) + selection + '_' + iteration
        final_stats_file = os.path.join(fingerprint, FINAL_STATS + '_' + experiment_name) + '.json'
        stats_file = os.path.join(fingerprint, STATS + '_' + experiment_name) + '.json'
        ginis_file = os.path.join(fingerprint, STATS_GINIS + '_' + experiment_name) + '.json'
        meta_file = os.path.join(fingerprint, 'META' + '_' + experiment_name) + '.json'
        graph_file = os.path.join(fingerprint, 'NETWORK' + '_' + experiment_name)
        flow_graph_file = os.path.join(fingerprint, 'FLOW' + '_' + experiment_name)

        G = nx.read_gpickle(graph_file)
        N = cls(G)
        N.fingerprint = fingerprint
        N.flow = nx.read_gpickle(flow_graph_file)
        with open(stats_file, "r") as f:
            stats = json.load(f)
            N.__stats = stats
        with open(ginis_file, "r") as f:
            ginis = json.load(f)
            N.__history_gini = ginis
        with open(meta_file, "r") as f:
            meta_info = json.load(f)
            N.__set_meta_info(meta_info)
        with open(final_stats_file, "r") as f:
            final_stas = json.load(f)
            N.__set_final_stats(final_stas)
        return N

    @classmethod
    def parse_clightning(cls, channel_file):
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
        diam = nx.diameter(T)
        logger.info('Max strongly connected component has diameter {}'.format(diam))


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

        return cls(G)

    @staticmethod
    def gini(x):
        # FIXME: replace with a more efficient implementation
        mean_absolute_differences = np.abs(np.subtract.outer(x, x)).mean()
        relative_absolute_mean = mean_absolute_differences / np.mean(x)
        return 0.5 * relative_absolute_mean

    @staticmethod
    def weighted_gini(x, weights=None):
        if weights is None:
            weights = np.ones_like(x)
        count = np.multiply.outer(weights, weights)
        mad = np.abs(np.subtract.outer(x, x) * count).sum() / count.sum()
        rmad = mad / np.average(x, weights=weights)
        return 0.5 * rmad

    # @staticmethod
    # def plot_gini_vs_rebalops_merge(networks, filename='gini_vs_rebalops_merge'):
    #     assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
    #     fingerprint = networks[0].fingerprint
    #     for n in networks:
    #         hist = n.gini_hist_data()
    #         lbl = '{:d} % participation'.format(int(n.participation * 100))
    #         plt.plot(hist, label=lbl, linewidth=3)
    #     plt.title("Network imbalance over time (successfull rebalancing operations)")
    #     plt.xlabel("Number of successfull rebalancing operations (logarithmic)")
    #     plt.ylabel("Network imbalance (G)")
    #     plt.xscale("log")
    #     plt.xlim(100, 10000000)
    #     plt.grid()
    #     plt.legend(loc="upper right")
    #     Network.__store_chart(fingerprint, filename)
    #
    # @staticmethod
    # def plot_paymentsize_vs_imbalance_merge(networks, filename='median_payment_size_merge'):
    #     # plot stats per 0.01 gini
    #     assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
    #     fingerprint = networks[0].fingerprint
    #     for n in networks:
    #         lbl = '{:d} % participation'.format(int(n.participation * 100))
    #         stats = n.stats()
    #         plt.plot(list(stats.keys())[::-1], [s['median_payment_amount'] for s in stats.values()][::-1],
    #                  label=lbl, linewidth=3)
    #         #plt.plot(list(stats.keys())[::-1], [s['median_payment_amount_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
    #     plt.title("Comparing Network imbalance with possible payment size")
    #     plt.xlabel("Network imbalance (G)")
    #     plt.ylabel("Median possible payment size [satoshi]")
    #     plt.legend(loc="upper right")
    #     plt.grid()
    #     Network.__store_chart(fingerprint, filename)
    #
    # @staticmethod
    # def plot_successrate_vs_imbalance_merge(networks, filename='successrate_vs_imbalance_merge'):
    #     assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
    #     fingerprint = networks[0].fingerprint
    #     for n in networks:
    #         lbl = '{:d} % participation'.format(int(n.participation * 100))
    #         stats = n.stats()
    #         plt.plot(list(stats.keys())[::-1], [s['success_rate'] for s in stats.values()][::-1],
    #                  label=lbl, linewidth=3)
    #         #plt.plot(list(stats.keys())[::-1], [s['success_rate_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
    #     plt.title("Comparing Network imbalance with success rate of random payments")
    #     plt.xlabel("Network imbalance (G)")
    #     plt.ylabel("Success rate of random payment")
    #     plt.grid()
    #     plt.legend(loc="lower left")
    #     Network.__store_chart(fingerprint, filename)
    #
    # @staticmethod
    # def plot_gini_vs_participation(networks, filename='gini_vs_participation'):
    #     def byParticipation(n):
    #         return n.participation
    #     assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
    #     fingerprint = networks[0].fingerprint
    #     networks.sort(key=byParticipation)
    #     data = [n.gini_distr_data() for n in networks]
    #     lbl = [str(n.participation*100)+'%' for n in networks]
    #     # Create a figure instance
    #
    #     plt.boxplot(data, labels=lbl)
    #     Network.__store_chart(fingerprint, filename)

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

    def debug(self):
        print('nodes: {}. all pair short path #{}'.format(len(self.G.nodes), len(self.__compute_all_pair_shortest_paths())))
