"""Lightning Network Simulation.

Usage:
  simulate.py run (import|restore) <filename> <participation> [--force]
  simulate.py runmulti (import|restore) <filename>
  simulate.py -h | --help

Options:
  -h --help     Show this screen.
  --version     Show version.
  --speed=<kn>  Speed in knots [default: 10].
  --force       Do not ask for every step.
  --drifting    Drifting mine.


"""
import sys
from docopt import docopt
import logging
# disable matplotlib logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

FORMAT ='%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format = FORMAT)
logger = logging.getLogger(__name__)

import os
from Network import Network, CYCLES_FILE
import time

def cont(question):
    print(question)
    answer = input("Do you want to proceed? (y / n): ")
    if not answer.lower() == 'y':
        return False
    return True

def init_network(arguments):
    if arguments["import"]:
        start = time.time()
        n = Network.parse_clightning(arguments["<filename>"], int(arguments["<participation>"])/100)
        n.create_snapshot()
        end = time.time()
        logger.info("Time to init network {0:.2f} seconds".format(end - start))
        return n
    elif arguments["restore"]:
        start = time.time()
        n = Network.restore_snapshot(arguments["<filename>"], int(arguments["<participation>"])/100)
        end = time.time()
        logger.info("Time to init network {0:.2f} seconds".format(end - start))
        return n

def compute_circles(n):
    start = time.time()
    n.compute_circles()
    end = time.time()
    logger.info("Time to calculate rebalancing circles {0:.2f} seconds".format(end - start))

def rebalance(n):
    start = time.time()
    n.rebalance(max_ops=100000, amount_coeff=1)
    n.store_experiment_result()
    end = time.time()
    logger.info("Time to rebalance the network {0:.2f} seconds".format(end - start))

def create_charts(n):
    n.plot_gini_distr_hist()
    n.plot_gini_vs_rebalops()
    n.plot_paymentsize_vs_imbalance()
    n.plot_successrate_vs_imbalance()

def main(arguments):
    force = arguments["--force"]
    if arguments["run"]:
        n = init_network(arguments)
        n.plot_gini_distr_hist('initial_gini_distr_hist')
        logger.info(n)
        if not force and not cont('Next step: Compute rebalancing circles.'):
            exit()
        compute_circles(n)
        if not force and not cont('Next step: Rebalance the network.'):
            exit()
        rebalance(n)
        if not force and not cont('Next step: Create charts from the previous rebalancing operations.'):
            exit()
        create_charts(n)

    if arguments["runmulti"]:
        experiments = [(i+1) * 10 for i in range(10)][::-1]
        #experiments = [100,50]
        sample_size = 1
        networks = dict()
        samples = dict()
        for exp in experiments:
            for i in range(sample_size):
                try:
                    n = Network.restore_result(arguments["<filename>"], exp / 100, iteration=i+1, selection='random')
                    print('Existing result found for experiment {}, iteration {}'.format(exp, i+1))
                except:
                    print('No result found for experiment {}, iteration {}. Execute experiment now.'.format(exp, i+1))
                    n = Network.restore_snapshot(arguments["<filename>"], exp / 100, iteration=i+1, selection='random')
                    n.plot_gini_distr_hist('initial_gini_distr_hist')
                    compute_circles(n)
                    rebalance(n)
                samples[i] = n
            # condense the networks
            cond = Network.condense_networks(list(samples.values()))
            networks[exp] = cond
        # some plots only use a subset of participating networks
        subset = [n for n in networks.values()]# [networks[100], networks[90], networks[80], networks[50]] #

        Network.plot_gini_vs_rebalops_merge(subset)
        Network.plot_paymentsize_vs_imbalance_merge(subset)
        Network.plot_successrate_vs_imbalance_merge(subset)
        Network.plot_gini_vs_participation(subset)

if __name__ == '__main__':
    # print('correct args', sys.argv)
    arguments = docopt(__doc__, version='Lightning Network Simulation 1.0')
    main(arguments)
