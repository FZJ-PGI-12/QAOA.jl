# [Sherrington–Kirkpatrick model](@id SK)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/sherrington_kirkpatrick.ipynb)
    with this example is available in the [notebooks folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The Sherrington–Kirkpatrick (SK) model is a spin glass: every pair of spins
interacts through a random coupling. The task is to find the spins of lowest
energy

```math
E = \frac{1}{\sqrt{n}} \sum_{i<j} J_{ij} \sigma_i \sigma_j,
```

where the couplings ``J_{ij}`` are independent standard normal numbers.
[`sherrington_kirkpatrick`](@ref) draws an instance with 12 spins:

```@example sk
using QAOA, Random
n = 12
sk = sherrington_kirkpatrick(n; seed = 1)
```

For 12 spins, a search through all 4096 assignments finds the lowest energy:

```@example sk
spins(b) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:n]
ground_energy = minimum(objective(sk, spins(b)) for b in 0:2^n-1)
```

## Annealing

With a [`linear_schedule`](@ref) of `p` layers, [`solve`](@ref) runs a
quantum annealing and measures the result 1000 times:

```@example sk
for p in (10, 50)
    result = solve(sk, linear_schedule(p); rng = Xoshiro(1))
    println("p = $p: average energy ", round(result.expectation; digits = 3),
            ", lowest measured energy ", round(result.objective; digits = 3),
            ", measured with probability ", round(result.probability; digits = 3))
end
```

Both runs find the ground state. With 50 layers, a quarter of all
measurements return it.

## Optimized angles

Three optimized layers:

```@example sk
tuned, energy = optimize_schedule(sk, 3)
tuned
```

```@example sk
energy
```

The SK energy does not change when all spins flip, so the QAOA state has the
same symmetry. The workspace stores half of the state, which halves memory
and time:

```@example sk
Workspace(sk)
```
