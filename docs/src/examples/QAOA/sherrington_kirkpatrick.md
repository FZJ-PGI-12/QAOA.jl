# [Sherrignton-Kirkpatrick Model] (@id SKModel)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/sherrington_kirkpatrick.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The cost function of the SK model is defined as

```math
\begin{align*}
\hat H_P = \frac{1}{\sqrt{N}}\sum_{i<j\leq N} J_{ij} \hat{Z}_i \hat{Z}_j,
\end{align*}
```

where the couplings ``J_{ij}`` are i.i.d. standard Gaussian variables, i.e. with zero mean ``\left\langle J_{ij} \right\rangle = 0`` and variance ``\left\langle J_{ij}^2 \right\rangle = J^2``.

We can set this model up as follows:
```julia
using QAOA, LinearAlgebra
import Random, Distributions

N = 4
σ2 = 1.0

Random.seed!(1)
J = rand(Distributions.Normal(0, σ2), N, N) ./ sqrt(N) 
J[diagind(J)] .= 0.0
J = UpperTriangular(J)
J = J + transpose(J)
```
We have two options to get the corresponding [`Problem`](@ref) from `QAOA.jl`. We can pass the coupling matrix ``J`` directly:
```julia
p = 2
SK_problem = QAOA.Problem(p, zeros(N), J)
```
or we can use a predefined wrapper function that constructs ``J`` from the above parameters and directly returns a [`Problem`](@ref):
```julia
SK_problem = QAOA.sherrington_kirkpatrick(N, σ2, num_layers=p, seed=1)
```
Given `SK_problem`, we can then call the gradient optimizer:
```julia
learning_rate = 0.02
cost, params, probs = QAOA.optimize_parameters(SK_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)
```
Alternatively, we can use [`NLsolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl):
```julia
cost, params, probs = QAOA.optimize_parameters(SK_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)
```