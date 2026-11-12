# [MaxCut](@id MaxCut)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/max_cut.ipynb)
    with this example is available in the [notebooks folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

MaxCut splits the nodes of a graph into two groups so that the edges between
the groups have the largest total weight. Spin `σᵢ = ±1` says which group
node `i` joins, and an edge `(i, j)` is cut when `σᵢ σⱼ = -1`. The weight of
the cut is therefore

```math
C = \sum_{(i, j) \in E} w_{ij} \, \frac{1 - \sigma_i \sigma_j}{2}.
```

The Petersen graph has 10 nodes and 15 edges, and its largest cut has 12
edges. [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) provides the
graph, and [`max_cut`](@ref) turns it into a problem:

```@example maxcut
using QAOA, Graphs, Random
petersen = smallgraph(:petersen)
problem = max_cut(petersen)
```

The same problem also follows from a list of edges, as in
`max_cut(10, [(1, 2), (1, 5), …])`. To count how often the QAOA finds a
largest cut, we need the cut of every assignment. Assignment `b` has
`σᵢ = +1` where bit `i - 1` of the integer `b` is 1:

```@example maxcut
spins(b) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:10]
cut_sizes = [objective(problem, spins(b)) for b in 0:2^10-1]
maximum(cut_sizes)
```

## Annealing

With [`linear_schedule`](@ref), the QAOA is a quantum annealing from the
mixer to the objective. More layers make the annealing slower and the result
better:

```@example maxcut
ws = Workspace(problem)
for p in (10, 30, 100)
    schedule = linear_schedule(p)
    P = probabilities(ws, schedule)
    println("p = $p: average cut ", round(expectation(ws, schedule); digits = 3),
            ", probability of a largest cut ", round(sum(P[cut_sizes .== 12]); digits = 3))
end
```

## Optimized angles

[`optimize_schedule`](@ref) tunes the angles of `p` layers, one layer at a
time: the optimum for `p - 1` layers, interpolated to `p` layers, starts the
search for `p` layers.

```@example maxcut
for p in 1:5
    tuned, value = optimize_schedule(ws, p)
    P = probabilities(ws, tuned)
    println("p = $p: average cut ", round(value; digits = 3),
            ", probability of a largest cut ", round(sum(P[cut_sizes .== 12]); digits = 3))
end
```

Four optimized layers find a largest cut about as often as 30 layers of
annealing. The optimized angles resemble an annealing schedule: `β` falls
from layer to layer, and `γ` mostly rises.

```@example maxcut
tuned, _ = optimize_schedule(ws, 4)
tuned
```

[`solve`](@ref) measures the final state and reports the best cut it saw:

```@example maxcut
result = solve(ws, tuned; rng = Xoshiro(1))
```

Flipping all spins leaves every cut unchanged, since the two groups only
swap names. The workspace uses this symmetry and stores only half of the
quantum state:

```@example maxcut
ws
```
