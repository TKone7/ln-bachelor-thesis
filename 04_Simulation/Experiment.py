from Network import Network, CYCLES_FILE, STATS_GINIS, STATS, MICRO_PAYMENT_SIZE, NORMAL_PAYMENT_SIZE
import networkx as nx
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

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

        # only for type random
        self.samplesize = None
        self.participations = None

    def setup_randomexperiment(self, samplesize=10):
        # self.participations = [(i + 1) * 10 for i in range(4,10)][::-1] # 10-100
        self.type = 'random'
        self.samplesize = samplesize
        self.participations = [100, 80]

    def run_experiment(self):
        if self.type == 'random':
            self.__execute_random()

    def plot_experiment(self):
        if self.type == 'random':
            self.__plot_random()

    def __execute_random(self):
        for exp in self.participations:
            for i in range(self.samplesize):
                if exp == 100 and i > 0:
                    print('No need to sample multiple times with 100% participation.')
                    continue
                try:
                    n = self.__load_existing_experiment(exp / 100, i + 1)
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

    def __plot_random(self):
        networks = dict()
        for exp in self.participations:
            samples = dict()
            for i in range(self.samplesize):
                if exp == 100 and i > 0:
                    print('No need to sample multiple times with 100% participation.')
                    continue
                try:
                    n = self.__load_existing_experiment(exp / 100, i + 1)
                except OSError as e:
                    logger.error('Cannot plot the defined experiment, as results are not there.')
                    raise
                samples[i] = n
            condensed = self.condense_networks(list(samples.values()))
            networks[exp] = condensed

        # subset = [n for n in networks.values()]
        for n in networks.values():
            n.plot_payments_vs_imbalance()

        subset = [value for key, value in networks.items() if key in (100, 90, 80, 60)]
        self.plot_gini_vs_rebalops(subset)
        self.plot_paymentsize_vs_imbalance(subset)
        self.plot_successrate_vs_imbalance(subset)
        # self.plot_gini_vs_participation(subset)
        self.plot_payments_vs_imbalance_one(subset)
        self.plot_payments_vs_imbalance_micro(subset)
        self.plot_payments_vs_imbalance_normal(subset)

    def __load_existing_experiment(self, participation, iteration):
        n = Network.restore_result(self.fingerprint, participation, iteration, self.type)
        n.set_experiment_name(str(participation * 100) + '_' + self.type + '_' + str(iteration))
        n.set_participation(participation)
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
            lbl = '{:d} % participation'.format(int(n.participation * 100))
            stats = n.stats()
            plt.plot(list(stats.keys())[::-1], [s['median_payment_amount'] for s in stats.values()][::-1],
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
        def byParticipation(n):
            return n.participation

        assert len(set([n.fingerprint for n in networks])) == 1, 'You cannot plot different networks together'
        fingerprint = networks[0].fingerprint
        networks.sort(key=byParticipation)
        data = [n.gini_distr_data() for n in networks]
        lbl = [str(n.participation * 100) + '%' for n in networks]
        # Create a figure instance

        plt.boxplot(data, labels=lbl)
        Experiment.__store_chart(fingerprint, filename + '_' + self.type)

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
        return N

e = Experiment('3a65a961')
e.setup_randomexperiment(1)
# e.run_experiment()
e.plot_experiment()