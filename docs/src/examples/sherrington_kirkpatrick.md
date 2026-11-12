# [Sherrington-Kirkpatrick Model](@id SK)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/sherrington_kirkpatrick.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The cost function of the SK model is defined as

```math
\begin{align*}
C(\sigma) = \frac{1}{\sqrt{N}}\sum_{i<j\leq N} J_{ij} \sigma_i \sigma_j,
\end{align*}
```

where the couplings ``J_{ij}`` are i.i.d. standard Gaussian variables, i.e. with zero mean ``\left\langle J_{ij} \right\rangle = 0`` and unit variance ``\left\langle J_{ij}^2 \right\rangle = 1``. The goal is to _minimize_ ``C``, i.e. to find the ground state of this spin glass.

We can set this model up as follows, with the factor ``1/\sqrt{N}`` included in `J`:
```@example sk
using QAOA, Random

N = 12
rng = Xoshiro(1)
J = zeros(N, N)
for j in 1:N, i in 1:j-1
    J[i, j] = J[j, i] = randn(rng) / sqrt(N)
end
```
We have two options to get the corresponding [`IsingProblem`](@ref) from `QAOA.jl`. We can pass the coupling matrix `J` directly:
```@example sk
SK_problem = IsingProblem(zeros(N), J)
```
or we can use a predefined wrapper function that constructs `J` in the same way and directly returns an [`IsingProblem`](@ref):
```@example sk
SK_problem = sherrington_kirkpatrick(N; seed = 1)
```
Note that `QAOA.jl` _minimizes_ the cost function by default, so here we do not need to pass a `sense`.

For 12 spins, we can still search through all 4096 bitstrings to find the lowest energy:
```@example sk
spins(b) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:N]
ground_energy = minimum(objective(SK_problem, spins(b)) for b in 0:2^N-1)
```

## Annealing

Given `SK_problem`, we can simulate a quantum annealing with a [`linear_schedule`](@ref) of `p` layers. [`solve`](@ref) then measures the final state 1000 times:
```@example sk
for p in (10, 50)
    result = solve(SK_problem, linear_schedule(p); rng = Xoshiro(1))
    println("p = $p: average energy ", round(result.expectation; digits = 3),
            ", lowest measured energy ", round(result.objective; digits = 3),
            ", measured with probability ", round(result.probability; digits = 3))
end
```
Both runs find the ground state. With 50 layers, a quarter of all measurements return it.

## Optimized Angles

Alternatively, we can optimize the angles of three layers with [`optimize_schedule`](@ref):
```@example sk
tuned, energy = optimize_schedule(SK_problem, 3)
tuned
```
```@example sk
energy
```

Like the cut of MaxCut, the energy of the SK model does not change when all spins flip, so the QAOA state has the same symmetry. The [`Workspace`](@ref) thus stores only half of the state, which halves memory and time:
```@example sk
Workspace(SK_problem)
```
