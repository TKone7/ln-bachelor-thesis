"""Lightning Network Simulation.

Usage:
  simulate.py parse <filename>
  simulate.py randomexperiment <fingerprint> <samplesize> [--charts]
  simulate.py bysize <fingerprint> [(asc|desc)] [--charts]
  simulate.py groupedsize <fingerprint> [--charts]
  simulate.py spread <fingerprint> <init> <spread> [--charts]
  simulate.py -h | --help

Options:
  -h --help     Show this screen.
  --charts      Skips the experiment and creates only the charts

"""
from Network import Network, CYCLES_FILE
from Experiment import Experiment
import logging
from docopt import docopt

# disable matplotlib logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


def main(args):
    if args["parse"]:
        n = Network.parse_clightning(args["<filename>"])
        n.create_snapshot()
        logger.info("The network {} can now be used for experiments.".format(n.fingerprint))

    if args["randomexperiment"]:
        e = Experiment(args["<fingerprint>"])
        e.setup_randomexperiment(int(args["<samplesize>"]))
        if not args["--charts"]:
            e.run_experiment()
        e.plot_experiment()
    if args["bysize"]:
        e = Experiment(args["<fingerprint>"])
        if args["asc"]:
            e.setup_experiment_by_size('asc')
        else:
            e.setup_experiment_by_size('desc')
        if not args["--charts"]:
            e.run_experiment()
        e.plot_experiment()
    if args["groupedsize"]:
        e = Experiment(args["<fingerprint>"])
        e.setup_experiment_by_size_grouped()
        if not args["--charts"]:
            e.run_experiment()
        e.plot_experiment()
    if args["spread"]:
        e = Experiment(args["<fingerprint>"])
        e.setup_experiment_netw_spread(int(args["<init>"]), int(args["<spread>"]))
        if not args["--charts"]:
            e.run_experiment()
        e.plot_experiment()


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Lightning Network Simulation 1.0')
    main(arguments)
