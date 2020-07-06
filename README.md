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
  - [ ]  _Optional_: Develop heuristic for intial channel balances
- [x] Reproduce simulation results from previous study
- [x] Simulate different levels of participation
  - [ ] _Optional_: Define different balance measurement
  - [ ] _Optional_: Define different success measurements
  - [ ] _Optional_: Non-random selection of participants
  - [ ] _Optional_: Measure Gini on participants, performance on total network
  - [ ] _Optional_: Measure Gini on participants, performance on participants
- [ ] Analyze and visualize results
