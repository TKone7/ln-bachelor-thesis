from Network import Network, MICRO_PAYMENT_SIZE, NORMAL_PAYMENT_SIZE
import networkx as nx
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from collections import Counter

# disable matplotlib logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

class Experiment:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.type = None
        self.participations = None

        # only for type random
        self.samplesize = None

        # only for experiment by size
        self.direction = None

        # only for experiment by spread
        self.init = None
        self.spread = None

        # only for grouped size
        self.bins = None

    def setup_experiment_netw_spread(self, init, spread):
        self.type = 'netw_spread_' + str(init).zfill(2) + '_' + str(spread).zfill(2)
        self.init = init
        self.spread = spread

    def setup_experiment_by_size(self, direction='desc'):
        self.type = 'size_linear' + '_' + direction
        self.participations = [(i + 1) * 10 for i in range(0, 10)][::-1] if direction == 'desc' else [100, 95, 90, 80,70,60,50,40,30,20,10]
        self.direction = direction

    def setup_experiment_by_size_grouped(self):
        self.type = 'size_category'
        self.bins = ['small', 'medium', 'large']
        self.participations = [(i + 1) * 10 for i in range(0, 10)][::-1]

    def setup_randomexperiment(self, samplesize=10):
        self.participations = [(i + 1) * 10 for i in range(0,10)][::-1] # 10-100
        self.type = 'random'
        self.samplesize = samplesize
        # self.participations = [100]

    def run_experiment(self):
        if self.type == 'random':
            self.__execute_random()
        elif self.type.startswith('size_linear'):
            self.__execute_size_linear()
        elif self.type.startswith('netw_spread_'):
            self.__execute_netw_spread()
        elif self.type.startswith('size_category'):
            self.__execute_size_category()

    def plot_experiment(self):
        if self.type == 'random':
            self.__plot_random()
        elif self.type.startswith('netw_spread_'):
            self.__plot_spread()
        elif self.type.startswith('size_linear'):
            self.__plot_size_linear()
        elif self.type.startswith('size_category'):
            self.__plot_size_category()
        else:
            raise NotImplementedError

    def __execute_size_category(self):
        def chunks(l, n):
            """Yield n number of sequential chunks from l."""
            d, r = divmod(len(l), n)
            for i in range(n):
                si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                yield l[si:si + (d + 1 if i < r else d)]
        for i, bin in enumerate(self.bins):
            # if i != 1:
            #     continue
            repeated_empty = 0
            for exp in self.participations:
                try:
                    n = self.__load_existing_experiment(exp, i + 1)
                    print('Existing result found for experiment {}'.format(exp))
                except OSError as e:
                    random.seed(100)
                    print('Execute experiment now. Size {}, participation of group {}%'.format(bin, exp))
                    n = Network.restore_snapshot(self.fingerprint)
                    n.set_experiment_name(str(exp) + '_' + self.type + '_' + str(i + 1))
                    n.set_participation(exp / 100)
                    # if not done yet, calculate the rebalancing circles
                    n.compute_circles()

                    degree_centrality_dict = nx.degree_centrality(n.G)
                    sorted_nodes = [k for k, _ in sorted(degree_centrality_dict.items(), key=lambda item: item[1])]
                    lists = list(chunks(sorted_nodes, 3))
                    logger.info('Threshold between {} and {} is {}. Between {} and {} is {}.'.format(self.bins[0], self.bins[1], n.G.in_degree(lists[1][0]), self.bins[1], self.bins[2], n.G.in_degree(lists[2][0])))
                    self.plot_channel_distr(n, [n.G.in_degree(lists[1][0]), n.G.in_degree(lists[2][0])])
                    current_set = lists[i]
                    incl = random.sample(current_set, int(len(current_set) * exp / 100))

                    excl = list(set(sorted_nodes) - set(incl))
                    n.exclude(excl)
                    n.rebalance(max_ops=100000, amount_coeff=1)
                    try:
                        n.store_experiment_result()
                    except AssertionError:
                        repeated_empty += 1
                        if repeated_empty > 1:
                            break
                        continue

    def __execute_netw_spread(self):
        all_participating = False
        iter = 1
        # n = Network.restore_snapshot(self.fingerprint)
        while not all_participating:
            print('Execute experiment now. Init {}% spread {}%.'.format(self.init, self.spread))
            n = Network.restore_snapshot(self.fingerprint)
            n.set_experiment_name(str(iter) + '_' + self.type + '_' + '1')
            n.set_participation(iter)
            # if not done yet, calculate the rebalancing circles
            n.compute_circles()
            # make selection here exclude certain nodes
            random.seed(100)
            nodes_sorted = list(n.G.nodes)
            nodes_sorted.sort()
            incl = random.sample(nodes_sorted, int(len(nodes_sorted) * (self.init / 100)))
            incl = set(incl)
            if iter > 1:
                for i in range(iter-1):
                    # add self.spread (%) of all neighbours to incl
                    neighbours = set()
                    neighbours_list = [{v for v in n.G[src]} for src in incl]
                    [neighbours.update(li) for li in neighbours_list]
                    random.seed((i+1) * 100)
                    neighbours_list = list(neighbours)
                    neighbours_list.sort()
                    selected_neighbours = random.sample(neighbours_list, int(len(neighbours_list) * (self.spread / 100)))
                    incl.update(selected_neighbours)

            logger.info('in round {} incl ({}) reached {}%'.format(iter, len(incl), len(incl)/len(nodes_sorted)*100))
            all_participating = int(len(incl)/len(nodes_sorted)*100) == 99
            excl = set(nodes_sorted) - incl
            n.exclude(excl)
            n.rebalance(max_ops=100000, amount_coeff=1)
            try:
                n.store_experiment_result()
            except AssertionError:
                logger.info('No experiment conducted since no routes found.')
            iter += 1

    def __execute_size_linear(self):
        for exp in self.participations:
            try:
                n = self.__load_existing_experiment(exp)
                print('Existing result found for experiment {}'.format(exp))
            except OSError as e:
                print('Execute experiment now.')
                n = Network.restore_snapshot(self.fingerprint)
                n.set_experiment_name(str(exp) + '_' + self.type + '_' + '1')
                n.set_participation(exp / 100)
                # if not done yet, calculate the rebalancing circles
                n.compute_circles()

                degree_centrality_dict = nx.degree_centrality(n.G)
                reverse = self.direction != 'desc'  # reverse the list because we select excluded nodes
                sorted_nodes = [k for k, _ in sorted(degree_centrality_dict.items(), key=lambda item: item[1], reverse=reverse)]

                excl = sorted_nodes[:round(len(sorted_nodes) * (1 - (exp / 100)))]
                n.exclude(excl)
                n.rebalance(max_ops=100000, amount_coeff=1)
                try:
                    n.store_experiment_result()
                except AssertionError:
                    continue

    def __execute_random(self):
        for exp in self.participations:
            for i in range(self.samplesize):
                if exp == 100 and i > 0:
                    print('No need to sample multiple times with 100% participation.')
                    continue
                try:
                    n = self.__load_existing_experiment(exp, i + 1)
                    print('Existing result found for experiment {}, iteration {}'.format(exp, i + 1))
                except OSError as e:
                    print('Execute experiment now.')
                    n = Network.restore_snapshot(self.fingerprint)
                    n.set_experiment_name(str(exp) + '_' + self.type + '_' + str(i + 1))
                    n.set_participation(exp / 100)
                    # if not done yet, calculate the rebalancing circles
                    n.compute_circles()
                    # make selection here exclude certain nodes
                    random.seed(i * 100)
                    nodes_sorted = list(n.G.nodes)
                    nodes_sorted.sort()
                    excl = random.sample(nodes_sorted, int(len(nodes_sorted) * (1 - (exp / 100))))
                    n.exclude(excl)
                    n.rebalance(max_ops=100000, amount_coeff=1)
                    n.store_experiment_result()

    def __plot_size_linear(self):
        networks = dict()
        for exp in self.participations:
            try:
                n = self.__load_existing_experiment(exp)
                networks[exp] = n
            except OSError as e:
                logger.error('Cannot plot the defined experiment, as results are not there.')
        net = list(networks.values())
        if self.direction == 'desc':
            subset = [value for key, value in networks.items() if key in (100, 80, 70, 50, 40)]
        else:
            subset = [value for key, value in networks.items() if key in (95, 90, 80, 70)]
        self.__plot_all(subset)
        self.__plot_vs_participation(net)

    def __plot_size_category(self):
        size = dict()
        for i, bin in enumerate(self.bins):
            networks = dict()
            for exp in self.participations:
                try:
                    n = self.__load_existing_experiment(exp, i + 1)
                    networks[exp] = n
                except OSError as e:
                    random.seed(100)
                    logger.error('Cannot plot the defined experiment, as results are not there.')
            size[bin] = networks


        net = size['large']
        subset = [value for key, value in net.items() if key in (100, 90, 80, 70)]
        all = list(net.values())
        self.__plot_all(subset)
        self.__plot_vs_participation(all)


    def __plot_spread(self):
        networks = dict()
        all_participating = False
        iter = 1
        # n = Network.restore_snapshot(self.fingerprint)
        while not all_participating:
            try:
                n = self.__load_existing_experiment(iter, '1')
                networks[iter] = n
            except OSError as e:
                logger.error('Cannot plot the defined experiment, as results are not there.')
                if len(networks) > 0:
                    all_participating = True
            iter += 1
        # IMPORTANT reverse the order to go from largest to smalles participation
        net = list(networks.values())
        self.__plot_vs_participation(net)
        # change order for certain polots
        net = net[::-1]

        # for these plots reduce the set
        net_small = net[0::2]
        self.__plot_all(net)

    def __plot_random(self):
        networks = dict()
        for exp in self.participations:
            samples = dict()
            for i in range(self.samplesize):
                if exp == 100 and i > 0:
                    print('No need to sample multiple times with 100% participation.')
                    continue
                try:
                    n = self.__load_existing_experiment(exp, i + 1)
                except OSError as e:
                    logger.error('Cannot plot the defined experiment, as results are not there.')
                samples[i] = n
            condensed = self.condense_networks(list(samples.values()))
            networks[exp] = condensed

        # subset = [n for n in networks.values()]
        # for n in networks.values():
        #     n.plot_payments_vs_imbalance()
        all = [value for key, value in networks.items()]
        subset = [value for key, value in networks.items() if key in (100, 90, 80, 70, 60, 50)]
        self.__plot_all(subset)
        self.__plot_vs_participation(all)

    def __plot(self):
        networks = dict()
        for exp in self.participations:
            try:
                n = self.__load_existing_experiment(exp)
                networks[exp] = n
            except OSError as e:
                logger.error('Cannot plot the defined experiment, as results are not there.')
        subset = [value for key, value in networks.items()]
        self.__plot_all(subset)

    def __plot_all(self, networks):
        self.plot_gini_vs_rebalops(networks)
        self.plot_paymentsize_vs_imbalance(networks)
        self.plot_successrate_vs_imbalance(networks)
        # self.plot_payments_vs_imbalance_one(networks)
        # self.plot_payments_vs_imbalance_micro(networks)
        # self.plot_payments_vs_imbalance_normal(networks)

    def __plot_vs_participation(self, networks):
        # new against participation in percent
        self.plot_paymentsize_vs_participation(networks)
        self.plot_successrate_vs_participation(networks)
        # self.plot_payments_vs_participation(networks)
        self.plot_gini_vs_participation(networks)

    def __load_existing_experiment(self, participation, iteration=1):
        experiment_name = self.fingerprint + '_' + str(int(participation)) + '_' + self.type + '_' + str(iteration)
        n = Network.restore_result_by_name(self.fingerprint, experiment_name)
        n.set_experiment_name(str(participation) + '_' + self.type + '_' + str(iteration))
        n.set_participation(participation)
        if self.type == 'random' or self.type.startswith('size_'):
            n.set_participation(participation / 100)
        else:
            n.set_participation((participation))
        return n

    def plot_gini_vs_rebalops(self, networks, filename='gini_vs_rebalops'):
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
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_paymentsize_vs_imbalance(self, networks, filename='median_payment_size'):
        # plot stats per 0.01 gini
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            if self.type.startswith('netw_spread_'):
                lbl = 'iteration {:d}'.format(int(n.participation))
            else:
                lbl = '{:d} % participation'.format(int(n.participation * 100))

            stats = n.stats()
            keys = list(stats.keys())
            k = keys[::-1]
            values = [s['median_payment_amount'] for s in stats.values()]
            v = values[::-1]
            plt.plot(k, v,
                     label=lbl, linewidth=3)
            # plt.plot(list(stats.keys())[::-1], [s['median_payment_amount_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
        plt.title("Comparing Network imbalance with possible payment size")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Median possible payment size [satoshi]")
        plt.legend(loc="upper right")
        plt.grid()
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_successrate_vs_imbalance(self, networks, filename='successrate_vs_imbalance'):
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['success_rate'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
            # plt.plot(list(stats.keys())[::-1], [s['success_rate_std'] for s in stats.values()][::-1], label=lbl + ' st dev', linewidth=3)
        plt.title("Comparing Network imbalance with success rate of random payments")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("Success rate of random payment")
        plt.grid()
        plt.legend(loc="lower left")
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_payments_vs_imbalance_one(self, networks, filename='payments_vs_imbalance_onesat'):
        raise DeprecationWarning
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['median_success'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
        plt.title("Comparing Network imbalance with ability to route 1")
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("median nr of payments (out of 10)")
        plt.grid()
        plt.legend(loc="lower left")
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_payments_vs_imbalance_micro(self, networks, filename='payments_vs_imbalance_micro'):
        raise DeprecationWarning
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['median_micro'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
        plt.title("Comparing Network imbalance with ability to route {}".format(MICRO_PAYMENT_SIZE))
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("median nr of payments (out of 10)")
        plt.grid()
        plt.legend(loc="lower left")
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_payments_vs_imbalance_normal(self, networks, filename='payments_vs_imbalance_normal'):
        raise DeprecationWarning
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        for n in networks:
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['median_normal'] for s in stats.values()][::-1],
                     label=lbl, linewidth=3)
        plt.title("Comparing Network imbalance with ability to route {}".format(NORMAL_PAYMENT_SIZE))
        plt.xlabel("Network imbalance (G)")
        plt.ylabel("median nr of payments (out of 10)")
        plt.grid()
        plt.legend(loc="lower left")
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_gini_vs_participation(self, networks, filename='gini_vs_participation'):

        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        k = []
        v = []
        error_up = []
        error_down = []
        node_participation = []
        capacity_participation = []
        for n in networks:
            participation = int(n.participation) if self.type.startswith('netw_spread_') else n.participation*100
            k.append(participation)

            if self.type == 'random':
                v.append(n.end_mean_gini)
                error_up.append(n.end_mean_gini + n.end_std_mean_gini)
                error_down.append(n.end_mean_gini - n.end_std_mean_gini)
            else:
                v.append(n.mean_gini)
            part_node = (n.nr_participating_nodes / n.nr_nodes) * 100
            part_capa = (n.incl_capacity / n.total_capacity) * 100
            node_participation.append(part_node)
            capacity_participation.append(part_capa)

        plt.plot(k, v, label='Mean Gini', linewidth=3)
        if self.type == 'random':
            plt.fill_between(k, error_down, error_up, color='moccasin')
        if self.type.startswith('netw_spread_'):
            plt.title("Comparing participation with mean gini\nInit: {}%, spread: {}%".format(self.init, self.spread))
            plt.xlabel("Iterations of spread")
            plt.xlim(1, 11)
        else:
            plt.title("Comparing participation with mean gini")
            plt.xlabel("Participation")
            plt.xlim(10, 100)

        plt.ylabel("Gini")
        plt.legend(loc='upper left')
        plt.grid(axis='y')

        ax2 = plt.twinx()
        if self.type.startswith('netw_spread_'):
            ax2.plot(k, node_participation, label='Node participation', linewidth=2, color='tab:purple', linestyle='dotted', marker='^')
        ax2.set_ylabel("Participation [%]")
        ax2.plot(k, capacity_participation, label='Capacity participation', linewidth=2, color='tab:cyan', linestyle='dotted', marker='o')
        ax2.legend(loc='lower left')

        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_paymentsize_vs_participation(self, networks, filename='median_paymnet_size_vs_participation'):
        # plot stats per 0.01 gini
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        k = []
        v = []
        node_participation = []
        capacity_participation = []
        for n in networks:
            participation = int(n.participation) if self.type.startswith('netw_spread_') else n.participation*100
            k.append(participation)
            stats = n.stats()
            values = [s['median_payment_amount'] for s in stats.values()]
            v.append(max(values))
            part_node = (n.nr_participating_nodes / n.nr_nodes) * 100
            part_capa = (n.incl_capacity / n.total_capacity) * 100
            node_participation.append(part_node)
            capacity_participation.append(part_capa)

        plt.plot(k, v, label='Payment size', linewidth=3)
        if self.type.startswith('netw_spread_'):
            plt.title("Comparing participation with possible payment size\nInit: {}%, spread: {}%".format(self.init, self.spread))
            plt.xlabel("Iterations of spread")
            plt.xlim(1, 11)
        else:
            plt.title("Comparing participation with possible payment size")
            plt.xlabel("Participation")
            plt.xlim(10, 100)

        plt.ylabel("Median possible payment size [satoshi]")
        plt.legend(loc='upper left')
        plt.grid(axis='y')

        ax2 = plt.twinx()
        if self.type.startswith('netw_spread_'):
            ax2.plot(k, node_participation, label='Node participation', linewidth=2, color='tab:purple', linestyle='dotted', marker='^')
        ax2.set_ylabel("Participation [%]")
        ax2.plot(k, capacity_participation, label='Capacity participation', linewidth=2, color='tab:cyan', linestyle='dotted', marker='o')
        ax2.legend(loc='lower left')

        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_successrate_vs_participation(self, networks, filename='successrate_vs_participation'):
        # plot stats per 0.01 gini
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        k = []
        v = []
        node_participation = []
        capacity_participation = []
        for n in networks:
            participation = int(n.participation) if self.type.startswith('netw_spread_') else n.participation*100
            k.append(participation)
            stats = n.stats()
            values = [s['success_rate'] for s in stats.values()]
            v.append(max(values))
            part_node = (n.nr_participating_nodes / n.nr_nodes) * 100
            part_capa = (n.incl_capacity / n.total_capacity) * 100
            node_participation.append(part_node)
            capacity_participation.append(part_capa)

        if self.type.startswith('netw_spread_'):
            plt.title("Comparing participation with success rate of random payments\nInit: {}%, spread: {}%".format(self.init, self.spread))
            plt.xlabel("Iterations of spread")
            plt.xlim(1, 11)
        else:
            plt.title("Comparing participation with success rate of random payments")
            plt.xlabel("Participation")
            plt.xlim(10, 100)
        plt.plot(k, v, label='Success rate', linewidth=3)
        plt.ylabel("Success rate of random payment")
        plt.legend(loc='upper left')
        plt.ylim(0, 1)
        plt.grid(axis='y')

        ax2 = plt.twinx()
        if self.type.startswith('netw_spread_'):
            ax2.plot(k, node_participation, label='Node participation', linewidth=2, color='tab:purple', linestyle='dotted', marker='^')
        ax2.set_ylabel("Participation [%]")
        ax2.plot(k, capacity_participation, label='Capacity participation', linewidth=2, color='tab:cyan', linestyle='dotted', marker='o')
        ax2.set_ylabel("Participation [%]")
        ax2.legend(loc='lower left')

        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_payments_vs_participation(self, networks, filename='median_paymnets_vs_participation'):
        # plot stats per 0.01 gini
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        k = []
        vsat = []
        vmicro = []
        vnormal = []
        node_participation = []
        capacity_participation = []
        for n in networks:
            participation = int(n.participation) if self.type.startswith('netw_spread_') else n.participation*100
            k.append(participation)
            stats = n.stats()
            micro = [s['median_micro'] for s in stats.values()]
            one = [s['median_success'] for s in stats.values()]
            normal = [s['median_normal'] for s in stats.values()]
            vsat.append(max(one))
            vmicro.append(max(micro))
            vnormal.append(max(normal))
            part_node = (n.nr_participating_nodes / n.nr_nodes) * 100
            part_capa = (n.incl_capacity / n.total_capacity) * 100
            node_participation.append(part_node)
            capacity_participation.append(part_capa)

        if self.type.startswith('netw_spread_'):
            plt.title("Comparing participation with possible payment size\nInit: {}%, spread: {}%".format(self.init, self.spread))
            plt.xlabel("Iterations of spread")
            plt.xlim(1, 11)
        else:
            plt.title("Comparing participation with possible payment size")
            plt.xlabel("Participation")
            plt.xlim(10, 100)
        plt.plot(k, vsat, label='Ability to route 1', linewidth=3)
        plt.plot(k, vmicro, label='Ability to route micropayment', linewidth=3)
        plt.plot(k, vnormal, label='Ability to route normal payment', linewidth=3)
        plt.ylim(0, 11)
        plt.ylabel("Median nr of payments (out of 10)")
        plt.legend(loc='upper left')
        plt.grid(axis='y')

        ax2 = plt.twinx()
        if self.type.startswith('netw_spread_'):
            ax2.plot(k, node_participation, label='Node participation', linewidth=2, color='tab:purple', linestyle='dotted', marker='^')
        ax2.set_ylabel("Participation [%]")
        ax2.plot(k, capacity_participation, label='Capacity participation', linewidth=2, color='tab:cyan', linestyle='dotted', marker='o')
        ax2.set_ylabel("Participation [%]")
        ax2.legend(loc='lower left')

        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

    def plot_channel_distr(self, network, boundaries, filename='channel_distribution'):
        file = filename
        for i in range(2):
            dict_degrees = dict(network.G.in_degree())
            degrees = list(dict_degrees.values())
            c = Counter(degrees)
            # logger.info(sorted(c.items()))
            plt.plot(*zip(*sorted(c.items())))

            # plt.hist(degrees, bins=100)
            if i == 1:
                plt.xscale("log")
                plt.axvline(x=boundaries[0], color='tab:orange', linestyle='dashed')
                plt.axvline(x=boundaries[1], color='tab:orange', linestyle='dashed')
                file += '_log_boundaries'
            plt.title("Distribution of nodes channelcount")
            plt.xlabel("Number of channels")
            plt.ylabel("Frequency of nodes")
            plt.grid()
            Experiment.__store_chart(self.fingerprint, file)

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

    def condense_networks(self, networks):
        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot condense different networks together'
        assert len(set([n.participation for n in networks])) == 1, 'You cannot condense different participations together'

        fingerprint = networks[0].fingerprint
        participation = networks[0].participation

        # Create a dummy network
        N = Network(nx.DiGraph())
        N.fingerprint = fingerprint
        N.set_participation(participation)
        N.set_experiment_name(str(int(participation * 100)) + '_' + self.type)

        all_keys = [set(n.stats().keys()) for n in networks]
        keys = set.intersection(*all_keys)

        figures = set()
        for n in networks:
            for v in list(n.stats().values()):
                figures.update(v.keys())

        key_l = list(keys)
        key_l.sort(reverse=True)
        print('keys {}, figures {}'.format(key_l, figures))

        avg_stats = dict()
        for k in key_l:
            avg_figures = dict()
            for f in figures:
                avg_figures[f] = np.average([n.stats()[k][f] for n in networks if k in n.stats()])
                avg_figures[f+'_std'] = np.std([n.stats()[k][f] for n in networks if k in n.stats()])
            avg_stats[k] = avg_figures
        N.set_stats(avg_stats)
        print(avg_stats)

        # gini history
        entries = min([len(n.gini_hist_data()) for n in networks])
        avg_gini_hist = [np.average([n.gini_hist_data()[e] for n in networks]) for e in range(entries)]
        N.set_history_gini(avg_gini_hist)

        # combine node / capacity statistics
        nr_participating_nodes = np.average([n.nr_participating_nodes for n in networks])
        nr_nodes = np.average([n.nr_nodes for n in networks])
        incl_capacity = np.average([n.incl_capacity for n in networks])
        total_capacity = np.average([n.total_capacity for n in networks])

        N.total_capacity = total_capacity
        N.incl_capacity = incl_capacity
        N.nr_participating_nodes = nr_participating_nodes
        N.nr_nodes = nr_nodes
        # Ginis

        N.end_mean_gini = np.average([n.mean_gini for n in networks])
        # N.end_std_mean_gini = np.average([np.std(n.gini_distr_data()) for n in networks])
        N.end_std_mean_gini = np.std([n.mean_gini for n in networks])
        return N