# [Minimum Vertex Cover] (@id MinVertexCover)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/min_vertex_cover.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks). See also [Wikipedia](https://en.wikipedia.org/wiki/Vertex_cover).

To be able to directly compare to the [Pennylane implementation](https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qaoa/cost.py), we employ the following cost function:

```math
\begin{align*}
    \hat C = -\frac 34 \sum_{(i, j) \in E(G)} (\hat Z_i \hat Z_j  +  \hat Z_i  +  \hat Z_j)  + \sum_{i \in V(G)} \hat Z_i,
\end{align*}
```

where ``E(G)`` is the set of edges and ``V(G)`` is the set of vertices of the graph ``G`` (we put a global minus sign since we _maximize_ the cost function).

We can set this model up as follows:
```julia
using QAOA, LinearAlgebra
import Random, Distributions
using PyCall
nx = pyimport("networkx")

N = 4
graph = nx.gnp_random_graph(N, 0.5, seed=7) 

h = -ones(N)
J = zeros(N, N)
for edge in graph.edges
    h[edge[1] + 1] += 3/4.
    h[edge[2] + 1] += 3/4.
    J[(edge .+ (1, 1))...] = 3/4.
end
```
We have two options to get the corresponding [`Problem`](@ref) from `QAOA.jl`. We can pass the coupling matrix `J` directly:
```julia
p = 2
mvc_problem = QAOA.Problem(p, -h, -J)
```
(since our algorithm _maximizes_ the cost function, we put in _extra minus signs_ for the problem parameters), or we can use a predefined wrapper function that constructs `J` from the above parameters and directly returns a [`Problem`](@ref):
```julia
mvc_problem = QAOA.min_vertex_cover(N, [edge .+ (1, 1) for edge in graph.edges], num_layers=p)
```
Given `mvc_problem`, we can then call the gradient optimizer:
```julia
learning_rate = 0.01
cost, params, probs = QAOA.optimize_parameters(mvc_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)
```
Alternatively, we can use [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl):
```julia
cost, params, probs = QAOA.optimize_parameters(mvc_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)
```