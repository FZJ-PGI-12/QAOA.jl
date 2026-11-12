# [Constraints and the XY Mixer](@id XY)

Some problems allow only bitstrings with a fixed number of ones. An example is a _bisection_ of a graph, i.e. a cut into two groups of equal size. We could enforce such a constraint with a penalty in the cost function, as for the [minimum vertex cover](@ref MVC). The [`XYMixer`](@ref) enforces it exactly instead: it moves amplitude only between bitstrings with the same number of ones. If we start the QAOA in [`dicke_state(n, k)`](@ref dicke_state), the equal superposition of all bitstrings with ``k`` ones, it thus never leaves these bitstrings.

As an example, we take a graph in which node 1 is joined to the nodes 2 to 6, and one more edge joins nodes 2 and 3. The largest cut puts node 1 alone into one group and cuts five edges:
```@example xy
using QAOA, Random
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3)]
graph = max_cut(6, edges)
solve(graph; p = 20, rng = Xoshiro(1)).objective
```
A bisection puts node 1 and two other nodes into one group, so it cuts three of the edges at node 1. It also cuts the edge between nodes 2 and 3 if these two nodes end up in different groups. The best bisection therefore cuts four edges.

The XY mixer acts on pairs of qubits. We choose the pairs along a ring, which connects all qubits, and start from the superposition of all bitstrings with three ones:
```@example xy
mixer = XYMixer([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)])
ws = Workspace(graph; mixer, initial_state = dicke_state(6, 3))
result = solve(ws, linear_schedule(20); rng = Xoshiro(1))
```
The best measured bitstring is a best bisection, with four cut edges:
```@example xy
result.x, result.objective
```
Note that all measured bitstrings have three ones, i.e. their probabilities add up to 1:
```@example xy
P = probabilities(ws, linear_schedule(20))
sum(P[b+1] for b in 0:63 if count_ones(b) == 3)
```
