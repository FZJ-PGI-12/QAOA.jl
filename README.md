<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.png">
    <img alt="QAOA.jl logo" src="docs/src/assets/logo.png" width="240">
  </picture>
</p>

# QAOA.jl

[![CI](https://github.com/FZJ-PGI-12/QAOA.jl/workflows/CI/badge.svg)](https://github.com/FZJ-PGI-12/QAOA.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/FZJ-PGI-12/QAOA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-PGI-12/QAOA.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fzj-pgi-12.github.io/QAOA.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://fzj-pgi-12.github.io/QAOA.jl/dev/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05364/status.svg)](https://doi.org/10.21105/joss.05364)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8086188.svg)](https://doi.org/10.5281/zenodo.8086188)

This package implements the [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028) (QAOA) efficiently, to support the extensive classical simulations typically required in research on the topic. `QAOA.jl` keeps the quantum state of all qubits in memory and thus gives exact probabilities and expectation values, together with their gradients with respect to the angles of the algorithm. One expectation value for 20 qubits takes a few milliseconds, and problems with about 30 qubits fit into the memory of a laptop.

> [!NOTE]
> **The mean-field AOA has moved to [MFAOA.jl](https://github.com/Quicopt/MFAOA.jl).** Up to version 1.4, `QAOA.jl` also implemented the [Mean-Field Approximate Optimization Algorithm](https://doi.org/10.1103/PRXQuantum.4.030335) (mean-field AOA). That algorithm now has its own package, `MFAOA.jl`, which also accepts QUBO problems and works as a JuMP optimizer. Both packages describe a problem and its angles in the same way. The Gaussian fluctuation analysis of the mean-field AOA (`evolve_fluctuations`) is available in `QAOA.jl` 1.4 and earlier.

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```

## Example

As an example, we take the MaxCut problem on a ring of five nodes, where at most four of the five edges can be cut:

```julia
using QAOA

ring = max_cut(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
result = solve(ring; p = 20)   # the QAOA with 20 layers, measured 1000 times
result.objective               # 4.0
```

`solve` simulates a quantum annealing cut into 20 layers. With few layers, `optimize_schedule` finds better angles. A number of common optimization problems such as MaxCut, the minimum vertex-cover problem, the Sherrington-Kirkpatrick model, and the partition problem are pre-implemented to facilitate scientific benchmarking. Beyond these, a problem can be an Ising model (`IsingProblem`), a QUBO (`QUBOProblem`), or any function of the bits (`GeneralProblem`).

## Documentation & Examples

Our docs can be found [here](https://fzj-pgi-12.github.io/QAOA.jl/dev/). Examples showcasing the use of `QAOA.jl` are also presented in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

## Performance

On the benchmark of Fig. 4 in [Golden et al. (2023)](https://arxiv.org/abs/2312.06451), MaxCut with 20 qubits and one layer, one expectation value takes 9.6 ms with `QAOA.jl` 2.0, 30 ms with JuliQAOA and 464 ms with `QAOA.jl` 1.4 (one thread of an Apple M4). Objectives that do not change when all bits flip, such as MaxCut, need only half the memory and time: 4.8 ms, or 1.7 ms with four threads. Our [performance page](https://fzj-pgi-12.github.io/QAOA.jl/dev/performance/) explains how.

## Support & Contributing

In case you need support or have encountered a problem with the package, you are welcome to [create an issue on GitHub](https://github.com/FZJ-PGI-12/QAOA.jl/issues). If you would like to contribute to `QAOA.jl`, you can reach us via [PGI-12](https://www.fz-juelich.de/en/pgi/pgi-12).


## Citations

If you are using code from this repository, please [cite our work](https://doi.org/10.21105/joss.05364).
