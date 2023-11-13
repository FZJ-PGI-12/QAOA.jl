# [MaxCut] (@id MaxCut)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/max_cut.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The cost function for the MaxCut problem as defined in the [original QAOA paper](https://arxiv.org/abs/1411.4028) is
```math
\begin{align*}
\hat C = \frac 12 \sum_{(i, j) \in E(G)} (1 - \hat Z_i \hat Z_j),
\end{align*}
```
where ``E(G)`` is the set of edges of the graph ``G``. 

We can set this model up as follows:
```julia
using QAOA, LinearAlgebra
import Random, Distributions
using PyCall
nx = pyimport("networkx");

N = 4
graph = nx.cycle_graph(N) 

h = zeros(N)
J = zeros(N, N)
for edge in graph.edges
    J[(edge .+ (1, 1))...] = -1/2.
end
```
Note that we have to _shift the edges by 1_ when going from Python to Julia.
We have two options to get the corresponding [`Problem`](@ref) from `QAOA.jl`. We can pass the coupling matrix `J` directly:
```julia
p = 1
max_cut_problem = QAOA.Problem(p, h, J)
```
or we can use a predefined wrapper function that constructs `J` from the above parameters and directly returns a [`Problem`](@ref):
```julia
max_cut_problem = QAOA.max_cut(N, [edge .+ (1, 1) for edge in graph.edges], num_layers=p)
```
Given `max_cut_problem`, we can then call the gradient optimizer:
```julia
learning_rate = 0.01
cost, params, probs = QAOA.optimize_parameters(max_cut_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)
```
Alternatively, we can use [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl):
```julia
cost, params, probs = QAOA.optimize_parameters(max_cut_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)
```
