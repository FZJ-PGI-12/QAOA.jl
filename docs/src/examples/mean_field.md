# [Mean-Field Approximate Optimization Algorithm] (@id MFAOA)

!!! tip
    For more details on the mean-field Approximate Optimization Algorithm, please consult [our paper](https://doi.org/10.1103/PRXQuantum.4.030335).

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/mean_field.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks). For a comparison between the QAOA and the mean-field AOA, have a look into our [SK](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/sherrington_kirkpatrick.ipynb) and [MaxCut](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/max_cut.ipynb) notebooks.


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

## Mean-Field Approximate Optimization

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
where we have taken into account that the _final spin is fixed_. The final vector of spins is then obtained via
```julia
evolve!(S, mf_problem.local_fields, mf_problem.couplings, β, γ)
```
The energy expectation value in mean-field approximation is 
```julia
E = expectation(S, mf_problem.local_fields, mf_problem.couplings)
```
and the solution of the algorithm can be retrieved by calling
```julia
sol = mean_field_solution(S)
```

## Equations of Motion via ODE Solver

There is also the option to solve the full mean-field equations of motion (s. e.g. equation (9) [in this paper](https://arxiv.org/pdf/2403.11548)) via `OrdinaryDiffEq.jl`. For a simple linear annealing schedule, we can do this by calling
```
T_final = p * τ
schedule_function = t -> t/T_final
sol = evolve(mf_problem.local_fields, mf_problem.couplings, T_final, schedule_function)
```

### Tensor Problem Definition

Instead of being restricted to simple QUBO problems defined by `local_fields` ``h_i`` and `couplings` ``J_{ij}`` as introduced above, we also want to be able to solve problems defined in terms of arbitrary tensors, e.g.
```math
\begin{align}
    H(t) = (1 - s(t)) \sum_{i=1}^N n_i^x(t) + s(t)\sum_{i=1}^N \bigg[ J_i + \sum_{j>i} \big[ J_{ij} + \sum_{k>j} J_{ijk}n_k^z(t) + \cdots \big]  n_j^z(t) \bigg] n_i^z(t).
\end{align}
```
To define the tensors for the standard QUBO set-up discussed here, we build the dictionaries
```julia
xtensor = Dict([(i, ) => 1.0 for i in 1:mf_problem.num_qubits])
```
for the local transverse-field driver and
```julia
ztensor = Dict()
for (i, h_i) in enumerate(mf_problem.local_fields)
    if h_i != 0.0
        ztensor[(i,)] = h_i
    end
end

for i in 1:mf_problem.num_qubits
    for j in i+1:mf_problem.num_qubits
        if mf_problem.couplings[i, j] != 0.0
            ztensor[(i, j)] = mf_problem.couplings[i, j]
        end
    end
end
```
for the problem Hamiltonian. We then feed these to
```julia
tensor_problem = TensorProblem(mf_problem.num_qubits, xtensor, ztensor)
```
and finally call
```julia
sol = evolve(tensor_problem, T_final, schedule_function)
```
which reproduces the same result as above.


!!! note
    `TensorProblem` can also deal with arbitrary higher-order tensors for the driver Hamiltonian!