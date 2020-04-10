# Effectivenes simulation of a rebalancing algorithm for the Lightning Network under partial participation

## Problem description

The Lightning Network is a second layer protocol in its infancy stage building on top of the Bitcoin network. It enables fast, secure, private, trustless, and permissionless payments without using the Bitcoin network for each transaction. Nodes of the network lock-up bitcoin into payment channels (edges) between them. Through those channels, payments can be routed between participants even without a direct edge.

A channel has a fixed capacity (determined during channel opening) which is distributed among the two channel partners. Payments can only be routed if all nodes along the path have enough local balances. For privacy reasons local channel balances are not public, which makes pathfinding in the Lightning Network a difficult problem.

A former paper proposes a change in the protocol that allows nodes to
rebalance their channels proactively with the objective to enable more and
larger payments and reducing errors during routing. While the previous
findings proved to be effective under full participation, adoption of such a protocol change in a decentralised network will be gradually and perhaps never reach 100%. Thereof, the question arises how those proposed
changes perform under a partial participation.

## Objective
- Definition of the scope of the protocol change
- Determine possible scenarios with varying participation
- Writing simulation software
- Collecting simulation data from the Lightning Network
- Running the simulation
- Evaluate and interpret the result
- Display results graphically
- Draw conclusions for the Lightning Network
