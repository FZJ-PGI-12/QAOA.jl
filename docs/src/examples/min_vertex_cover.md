# [Minimum Vertex Cover](@id MVC)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/min_vertex_cover.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks). It also compares with the cost function of the [Pennylane implementation](https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qaoa/cost.py). See also [Wikipedia](https://en.wikipedia.org/wiki/Vertex_cover).

A vertex cover of a graph is a set of nodes such that every edge has at least one end in the set. The minimum vertex cover has the fewest nodes. With bit ``x_i = 1`` for the chosen nodes, we employ the following cost function:

```math
\begin{align*}
C(x) = \sum_{i \in V(G)} x_i + P \sum_{(i, j) \in E(G)} (1 - x_i)(1 - x_j),
\end{align*}
```

where ``E(G)`` is the set of edges and ``V(G)`` is the set of vertices of the graph ``G``. The first sum counts the chosen nodes, and the second one adds the penalty ``P`` for every edge without a chosen end. With ``P > 1``, every minimum is a cover: covering a missing edge costs one node and saves at least ``P``. The goal is to _minimize_ ``C``.

Since ``C`` is a function of bits rather than spins, it is a QUBO. Expanding the products gives
```math
\begin{align*}
C(x) = P \sum_{(i, j) \in E(G)} x_i x_j + \sum_{i \in V(G)} (1 - P d_i) \, x_i + P |E(G)|,
\end{align*}
```
where ``d_i`` is the number of edges at node ``i``.

As an example, we take a graph in which nodes 1 and 5 each connect to nodes 2, 3 and 4. Nodes 1 and 5 together cover all six edges, and no other pair does. With ``P = 2``, we can set this model up as follows:
```@example mvc
using QAOA, Random

N = 5
edges = [(1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (5, 4)]
P = 2

Q = zeros(N, N)
c = ones(N)
for (i, j) in edges
    Q[i, j] += P
    c[i] -= P
    c[j] -= P
end
```
We have two options to get the corresponding [`QUBOProblem`](@ref) from `QAOA.jl`. We can pass the weights `Q` and `c` directly, together with the constant ``P |E(G)|``:
```@example mvc
mvc_problem = QUBOProblem(Q; c, constant = P * length(edges))
```
or we can use a predefined wrapper function that constructs `Q` and `c` from the edges and directly returns a [`QUBOProblem`](@ref), with ``P = 2`` by default:
```@example mvc
mvc_problem = min_vertex_cover(N, edges)
```
Given `mvc_problem`, we can then run a quantum annealing with 30 layers and measure the final state with [`solve`](@ref):
```@example mvc
result = solve(mvc_problem; p = 30, rng = Xoshiro(1))
```
The QAOA finds the cover of nodes 1 and 5:
```@example mvc
findall(==(1), result.x)
```
With `Graphs.jl` loaded, `min_vertex_cover(g)` takes a graph `g` instead.
