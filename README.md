# Effectivenes simulation of a rebalancing algorithm for the Lightning Network under partial participation
## Latest Version
I am in the process of writing the Thesis now. If you want to help me out you are free to read the current verstion here: https://github.com/TKone7/ln-bachelor-thesis/blob/master/06_Thesis/thesis.pdf
Just send me your feedback either per email, Github issue or pull request. Thank you very much for your collaboration.

## Problem description

The Lightning Network is a second layer protocol in its infancy stage building on top of the Bitcoin network. It enables fast, secure, private, trustless, and permissionless payments without using the Bitcoin network for each transaction. Nodes of the network lock-up bitcoin into payment channels (edges) between them. Through those channels, payments can be routed between participants even without a direct edge.

A channel has a fixed capacity (determined during channel opening) which is distributed among the two channel partners. Payments can only be routed if all nodes along the path have enough local balances. For privacy reasons local channel balances are not public, which makes pathfinding in the Lightning Network a difficult problem.

A former paper proposes a change in the protocol that allows nodes to
rebalance their channels proactively with the objective to enable more and
larger payments and reducing errors during routing. While the previous
findings proved to be effective under full participation, adoption of such a protocol change in a decentralised network will be gradually and perhaps never reach 100%. Thereof, the question arises how those proposed
changes perform under a partial participation.

## Main Objective
- [x] Rebuild the Lightning Network
  - [ ]  ~_Optional_: Develop heuristic for intial channel balances~
- [x] Reproduce simulation results from previous study
- [x] Simulate different levels of participation
  - [ ] ~_Optional_: Define different balance measurement~
  - [x] _Optional_: Define different success measurements
  - [x] _Optional_: Non-random selection of participants
  - [ ] ~_Optional_: Measure Gini on participants, performance on total network~
  - [ ] ~_Optional_: Measure Gini on participants, performance on participants~
- [x] Analyze and visualize results
- [ ] Write thesis (_in progress_)

## Run simulations yourself
1. Setup virtual environment and install dependencies
```bash
$ git clone git@github.com:TKone7/ln-bachelor-thesis.git
$ cd ln-bachelor-thesis/04_Simulation/
$ python3 -v venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
2. Use `channels.json` file from repo or use your own channel file to setup a network. You can use <a href="https://github.com/TKone7/clightning-plugins/tree/master/dumpgraph" target="_blank">this plugin</a> for creating the file with your c-lightning node.
```bash
$ python3 simulate.py parse channels.json
> The network 3a65a961 can now be used for experiments.
```
Use this **fingerprint** `3a65a961` for all subsequent experiments. It is unique for your set of nodes, channels, capacities/**balances** and fees.

3. You can now use this network to perform any experiment. **IMPORTANT**: The first experiment you will run pre-calculates a lot of things that might take a while. Subsequent experiments will run much faster.

### Random node selection
Selects the participating nodes at random. Executes all levels of participation from 100% to 10%.

Parameter:
- Fingerprint: Network fingerprint generated in previous step
- Samplesize: Executes the experiment repeatedly to remove variance.
- _Optional_ Charts: If selected, skips the experiment and plots charts from existing experiment results.

_Example_:
```bash
$ python3 simulate.py randomexperiment 3a65a961 1
$ python3 simulate.py randomexperiment 3a65a961 1 --charts
```
### Node selection by size
Orders nodes by degree (channel count) and simulates participation levels from either or the other end of the list.
Parameter:
- Fingerprint: Network fingerprint generated in previous step
- _Optional_: Direction (asc|__desc__): Participating nodes are chosen from the large end (desc) or the smaller end (asc)
- _Optional_ Charts: If selected, skips the experiment and plots charts from existing experiment results.

_Example_:
```bash
$ python3 simulate.py bysize 3a65a961 desc
$ python3 simulate.py bysize 3a65a961 desc --charts
```

### Node selection network spread
Defines a set of nodes that participate in iteration 1. With every iteration a certain percentage of adjacent nodes participate as well. Runs experiment until 99% of network is participating.
Parameter:
- Fingerprint: Network fingerprint generated in previous step
- Init: % of nodes participating initially
- Spread: % of adjacent nodes being added in every iteration
- _Optional_ Charts: If selected, skips the experiment and plots charts from existing experiment results.

_Example_:
```bash
$ python3 simulate.py spread 3a65a961 2 40
$ python3 simulate.py spread 3a65a961 2 40 --charts
```
