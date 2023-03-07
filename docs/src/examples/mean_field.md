# [Mean-Field Approximate Optimization Algorithm] (@id MFAOA)

!!! tip
    For more details on the mean-field Approximate Optimization Algorithm, please consult [our paper](https://arxiv.org/abs/2303.00329).

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/mean_field.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks). For a comparison between the QAOA and the mean-field AOA, have a look into our [MaxCut example](@ref MaxCut).


In close analogy to the QAOA, the mean-field Hamiltonian reads
```math
\begin{align}
    H(t) ={} & \gamma(t)\sum_{i=1}^N \bigg[ h_i + \sum_{j>i} J_{ij}  n_j^z(t) \bigg] n_i^z(t) + \beta(t) \sum_{i=1}^N n_i^x(t).
\end{align}
```
The mean-field evolution is then given by
```math
\begin{align}
    \boldsymbol{n}_i(p) = \prod_{k=1}^p \hat V_i^D(k) \hat V_i^P(k) \boldsymbol{n}_i(0),
\end{align}
```
where the initial spin vectors are ``\boldsymbol{n}_i(0) = (1, 0, 0)^T \; \forall i``, and the rotation matrices ``\hat V_i^{D,\,P}`` are defined as
```math
\begin{align}
\hat V_i^D(k) = 
\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos(2 \Delta_i\beta_k) & -\sin(2 \Delta_i \beta_k) \\
0 & \sin (2 \Delta_i \beta_k) & \phantom{-}\cos(2 \Delta_i \beta_k) 
\end{pmatrix}
\end{align}
```
and
```math
\begin{align}
\hat V_i^P(k) = 
\begin{pmatrix}
\cos(2m_i (t_{k-1}) \gamma_k) & -\sin(2m_i (t_{k-1}) \gamma_k) & 0 \\
\sin (2m_i (t_{k-1}) \gamma_k) & \phantom{-}\cos(2m_i (t_{k-1}) \gamma_k) & 0 \\
0 & 0 & 1
\end{pmatrix},
\end{align}
```
with the magnetization 
```math
\begin{align}
m_i(t) = h_i + \sum_{j=1}^N J_{ij} n_j^z(t).
\end{align}
```

To implement these dynamics within `QAOA.jl`, we begin by defining a schedule:
```julia
using QAOA, LinearAlgebra
import Random, Distributions

p = 100
τ = 0.5
γ = τ .* ((1:p) .- 1/2) ./ p |> collect
β = τ .* (1 .- (1:p) ./ p) |> collect
β[p] = τ / (4 * p)
```
Next, we set up a random instance of the Sherrington-Kirkpatrick model:
```julia
N = 5
σ2 = 1.0

Random.seed!(1)
J = rand(Distributions.Normal(0, σ2), N, N) ./ sqrt(N) 
J[diagind(J)] .= 0.0
J = UpperTriangular(J)
J = J + transpose(J)
```
Then we are ready to construct a [`Problem`](@ref):
```julia
mf_problem = Problem(p, J)
```

!!! note
    When the constructor for [`Problem`](@ref) is called _without_ `local_fields`, then `QAOA.jl` will automatically break the ``\mathcal{Z}_2`` symmetry of the underlying Hamiltonian. That is, the final spin will be held fixed, which automatically introduces effective `local_fields`. While this is _not required_ for the QAOA, it is a necessary preliminary for the mean-field AOA, since the mean-field dynamics will otherwise be trivial. 

This is all we need to call the mean-field dynamics. The initial values are
```julia
S = [[1., 0., 0.] for _ in 1:N-1]
```
where we have taken into account that the _final spin is fixed_. The final vector of spins is then obtained as
```julia
S = evolve!(S, mf_problem.local_fields, mf_problem.couplings, β, γ)
```
The energy expectation value in mean-field approximation is 
```julia
E = expectation(S[end], mf_problem.local_fields, mf_problem.couplings)
```
and the solution of the algorithm can be retrieved by calling
```julia
sol = mean_field_solution(S[end])
```