# [Partition Problem] (@id PartitionProblem)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/partition_problem.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The partition problem (see also [Wikipedia](https://en.wikipedia.org/wiki/Partition_problem)) for a set of uniformly distributed numbers ``\mathcal{S} = \{a_1, ..., a_N\}`` consists of finding two subsets `` \mathcal{S}_{1} \cup \mathcal{S}_2 =  \mathcal{S}`` such that the difference of the sums over the two subsets ``\mathcal{S}_{1, 2}`` is as small as possible. The cost function in Ising form can be defined as 
```math
\begin{align*}
\hat C = -\left(\sum_{i=1}^{N} a_i \hat{Z}_i\right)^2 = \sum_{i<j\leq N} J_{ij} \hat{Z}_i \hat{Z}_j + \mathrm{const.}
\end{align*}
```
with ``J_{ij}=-2a_i a_j``. The goal is then to _maximize_ ``\hat C``.

We can set this model up as follows:
```julia
using QAOA, LinearAlgebra
import Random, Distributions

N = 4
Random.seed!(1)
a = rand(Distributions.Uniform(0, 1), N)

J = -2 .* (a * transpose(a))
J[diagind(J)] .= 0.0
```
We have two options to get the corresponding [`Problem`](@ref) from `QAOA.jl`. We can pass the coupling matrix ``J`` directly:
```julia
p = 4
partition_problem = QAOA.Problem(p, zeros(N), J)
```
or we can use a predefined wrapper function that constructs ``J`` from the above parameters and directly returns a [`Problem`](@ref):
```julia
partition_problem = QAOA.partition_problem(a, num_layers=p)
```
Given `partition_problem`, we can then call the gradient optimizer:
```julia
learning_rate = 0.05
cost, params, probs = QAOA.optimize_parameters(partition_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)
```
Alternatively, we can use [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl):
```julia
cost, params, probs = QAOA.optimize_parameters(partition_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)
```
