# QAOA.jl

[![CI](https://github.com/FZJ-PGI-12/QAOA.jl/workflows/CI/badge.svg)](https://github.com/FZJ-PGI-12/QAOA.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/FZJ-PGI-12/QAOA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-PGI-12/QAOA.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fzj-pgi-12.github.io/QAOA.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://fzj-pgi-12.github.io/QAOA.jl/dev/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05364/status.svg)](https://doi.org/10.21105/joss.05364)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8086188.svg)](https://doi.org/10.5281/zenodo.8086188)

`QAOA.jl` simulates the [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028) (QAOA) on a classical computer. It keeps the quantum state of all qubits in memory and applies each layer of the algorithm to it directly. It therefore gives exact probabilities and expectation values, together with their derivatives with respect to the angles of the algorithm. One expectation value for 20 qubits takes a few milliseconds, and problems with about 30 qubits fit into the memory of a laptop.

> [!NOTE]
> **The mean-field AOA has moved to [MFAOA.jl](https://github.com/Quicopt/MFAOA.jl).** Up to version 1.4, QAOA.jl also implemented the [Mean-Field Approximate Optimization Algorithm](https://doi.org/10.1103/PRXQuantum.4.030335) (mean-field AOA). That algorithm now has its own package, MFAOA.jl, which also accepts QUBO problems and works as a JuMP optimizer. Both packages describe a problem and its angles in the same way. The Gaussian fluctuation analysis of the mean-field AOA (`evolve_fluctuations`) is available in QAOA.jl 1.4 and earlier.

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```

## Example

MaxCut splits the nodes of a graph into two groups so that as many edges as possible join the groups. On a ring of five nodes, at most four of the five edges can be cut:

```julia
using QAOA

ring = max_cut(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
result = solve(ring; p = 20)   # the QAOA with 20 layers, measured 1000 times
result.objective               # 4.0
```

`solve` runs a quantum annealing cut into 20 layers. With few layers, `optimize_schedule` finds better angles. A problem can be an Ising model (`IsingProblem`), a QUBO (`QUBOProblem`), or any function of the bits (`GeneralProblem`).

## Documentation & Examples

The [documentation](https://fzj-pgi-12.github.io/QAOA.jl/dev/) explains the algorithm and the package, with examples for MaxCut, the Sherrington–Kirkpatrick model, number partitioning, vertex cover and constrained problems. Jupyter notebooks are in the [notebooks folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

## Performance

On the benchmark of Fig. 4 in [Golden et al. (2023)](https://arxiv.org/abs/2312.06451), MaxCut with 20 qubits and one layer, one expectation value takes 9.6 ms with QAOA.jl 2.0, 30 ms with JuliQAOA and 464 ms with QAOA.jl 1.4 (one thread of an Apple M4). Objectives that do not change when all bits flip, such as MaxCut, need only half the memory and time: 4.8 ms, or 1.7 ms with four threads. The [performance page](https://fzj-pgi-12.github.io/QAOA.jl/dev/performance/) explains how.

## Support & Contributing

In case you need support or have encountered a problem with the package, you are welcome to [create an issue on GitHub](https://github.com/FZJ-PGI-12/QAOA.jl/issues). If you would like to contribute to `QAOA.jl`, you can reach us via [PGI-12](https://www.fz-juelich.de/en/pgi/pgi-12).


## Citations

If you are using code from this repository, please [cite our work](https://doi.org/10.21105/joss.05364).
