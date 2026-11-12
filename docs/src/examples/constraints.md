# [Constraints and the XY mixer](@id XY)

Some problems allow only assignments with a fixed number of ones, such as a
bisection: a split of a graph into two groups of equal size. A penalty in the
objective can enforce such a constraint. The [`XYMixer`](@ref) enforces it
exactly instead: it moves amplitude only between assignments with the same
number of ones. Started in [`dicke_state(n, k)`](@ref dicke_state), the equal
superposition of all assignments with `k` ones, the QAOA then never leaves
these assignments.

In the graph below, node 1 is joined to the nodes 2 to 6, and one more edge
joins nodes 2 and 3. The largest cut puts node 1 alone into one group and
cuts five edges:

```@example xy
using QAOA, Random
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3)]
graph = max_cut(6, edges)
solve(graph; p = 20, rng = Xoshiro(1)).objective
```

A bisection puts node 1 and two other nodes into one group, so it cuts three
of the edges at node 1. It also cuts the edge between nodes 2 and 3 if these
two nodes end up in different groups. The best bisection therefore cuts four
edges.

The XY mixer acts on pairs of qubits. Pairs along a ring connect all qubits:

```@example xy
mixer = XYMixer([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)])
ws = Workspace(graph; mixer, initial_state = dicke_state(6, 3))
result = solve(ws, linear_schedule(20); rng = Xoshiro(1))
```

The best measured assignment is a best bisection, with four cut edges:

```@example xy
result.x, result.objective
```

All measured assignments have three ones:

```@example xy
P = probabilities(ws, linear_schedule(20))
sum(P[b+1] for b in 0:63 if count_ones(b) == 3)
```
