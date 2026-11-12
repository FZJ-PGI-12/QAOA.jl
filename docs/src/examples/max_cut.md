# [MaxCut](@id MaxCut)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/max_cut.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The cost function for the MaxCut problem as defined in the [original QAOA paper](https://arxiv.org/abs/1411.4028) is
```math
\begin{align*}
C(\sigma) = \frac 12 \sum_{(i, j) \in E(G)} (1 - \sigma_i \sigma_j),
\end{align*}
```
where ``E(G)`` is the set of edges of the graph ``G``. Spin ``\sigma_i = \pm 1`` says which of the two groups node ``i`` joins, and ``C`` counts the edges between the groups. The goal is to _maximize_ ``C``.

As an example, we take the Petersen graph from [`Graphs.jl`](https://github.com/JuliaGraphs/Graphs.jl). It has 10 nodes and 15 edges, and its largest cut has 12 edges. We can set this model up as follows:
```@example maxcut
using QAOA, Graphs, Random
graph = smallgraph(:petersen)
N = nv(graph)

h = zeros(N)
J = zeros(N, N)
for edge in edges(graph)
    J[src(edge), dst(edge)] = J[dst(edge), src(edge)] = -1/2
end
```
We have two options to get the corresponding [`IsingProblem`](@ref) from `QAOA.jl`. We can pass `h` and the coupling matrix `J` directly, together with the constant ``|E(G)|/2`` of ``C``:
```@example maxcut
max_cut_problem = IsingProblem(h, J; offset = ne(graph) / 2, sense = :max)
```
Note that `QAOA.jl` _minimizes_ the cost function by default, so for MaxCut we pass `sense = :max`. Alternatively, we can use a predefined wrapper function that constructs `J` from the graph and directly returns an [`IsingProblem`](@ref):
```@example maxcut
max_cut_problem = max_cut(graph)
```
Without `Graphs.jl`, the wrapper takes the number of nodes and a list of edges instead, as in `max_cut(10, [(1, 2), (1, 5), …])`.

To see how often the QAOA finds a largest cut, we compute the cut of every bitstring. In bitstring `b`, spin ``\sigma_i = +1`` where bit `i - 1` of the integer `b` is 1:
```@example maxcut
spins(b) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:N]
cut_sizes = [objective(max_cut_problem, spins(b)) for b in 0:2^N-1]
maximum(cut_sizes)
```

## Annealing

We are going to evaluate many angles for the same problem, so we first build a [`Workspace`](@ref). It holds the cost of every bitstring and the quantum state:
```@example maxcut
ws = Workspace(max_cut_problem)
```
Flipping all spins leaves every cut unchanged, since the two groups only swap names. The workspace uses this symmetry and stores only half of the quantum state.

Simulating quantum annealing with `QAOA.jl` is very simple. For a _linear_ annealing schedule with `p` layers, all we need is [`linear_schedule`](@ref). More layers make the annealing slower and the result better:
```@example maxcut
for p in (10, 30, 100)
    schedule = linear_schedule(p)
    P = probabilities(ws, schedule)
    println("p = $p: average cut ", round(expectation(ws, schedule); digits = 3),
            ", probability of a largest cut ", round(sum(P[cut_sizes .== 12]); digits = 3))
end
```

## Optimized Angles

Given `ws`, we can also optimize the angles of `p` layers with [`optimize_schedule`](@ref), which uses exact gradients. It adds one layer at a time: the optimum for `p - 1` layers, interpolated to `p` layers, is where the search for `p` layers starts.
```@example maxcut
for p in 1:5
    tuned, value = optimize_schedule(ws, p)
    P = probabilities(ws, tuned)
    println("p = $p: average cut ", round(value; digits = 3),
            ", probability of a largest cut ", round(sum(P[cut_sizes .== 12]); digits = 3))
end
```
Four optimized layers find a largest cut about as often as 30 layers of annealing. The optimized angles resemble an annealing schedule: ``\beta`` falls from layer to layer, and ``\gamma`` mostly rises.
```@example maxcut
tuned, _ = optimize_schedule(ws, 4)
tuned
```
Finally, [`solve`](@ref) measures the final state 1000 times and returns the best cut it saw:
```@example maxcut
result = solve(ws, tuned; rng = Xoshiro(1))
```
