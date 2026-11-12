# [Minimum vertex cover](@id MVC)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/min_vertex_cover.ipynb)
    with this example is available in the [notebooks folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

A vertex cover of a graph is a set of nodes such that every edge has at least
one end in the set. The minimum vertex cover has the fewest nodes. With
`xᵢ = 1` for the chosen nodes, [`min_vertex_cover`](@ref) minimizes

```math
C = \sum_i x_i + P \sum_{(i, j) \in E} (1 - x_i)(1 - x_j),
```

the number of chosen nodes plus the penalty `P` for every edge without a
chosen end. With `P > 1`, every minimum is a cover: covering a missing edge
costs one node and saves at least `P`. The default is `P = 2`. The problem
has bits instead of spins, so it is a [`QUBOProblem`](@ref).

In the graph below, nodes 1 and 5 each connect to nodes 2, 3 and 4. Nodes 1
and 5 together cover all six edges, and no other pair does:

```@example mvc
using QAOA, Random
edges = [(1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (5, 4)]
cover = min_vertex_cover(5, edges)
result = solve(cover; p = 30, rng = Xoshiro(1))
```

The QAOA finds this cover:

```@example mvc
findall(==(1), result.x)
```

With Graphs.jl loaded, `min_vertex_cover(g)` takes a graph `g` instead.
