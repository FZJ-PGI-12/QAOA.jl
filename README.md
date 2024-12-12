# QAOA.jl

[![CI](https://github.com/FZJ-PGI-12/QAOA.jl/workflows/CI/badge.svg)](https://github.com/FZJ-PGI-12/QAOA.jl/actions?query=workflow%3ACI)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fzj-pgi-12.github.io/QAOA.jl/stable/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05364/status.svg)](https://doi.org/10.21105/joss.05364)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8086188.svg)](https://doi.org/10.5281/zenodo.8086188)

This package implements the [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028) and the [Mean-Field Approximate Optimization Algorithm](https://link.aps.org/doi/10.1103/PRXQuantum.4.030335).

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```

## Documentation & Examples

Our docs can be found [here](https://fzj-pgi-12.github.io/QAOA.jl/dev/). Examples showcasing the use of `QAOA.jl` are also presented in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

## Benchmarks

`QAOA.jl` also supports gradient optimization via automatic differentiation. Below is a comparison of run times between `PennyLane` [@PennyLane] and `QAOA.jl` on an Apple M1 processor. The benchmarks are retrieved by performing 128 steps with the respective gradient optimizer on the same instance of size $N$ of the minimum vertex-cover problem.

<img src="https://raw.githubusercontent.com/FZJ-PGI-12/QAOA.jl/master/assets/benchmarks.png" align="middle"/>


## Support & Contributing

In case you need support or have encountered a problem with the package, you are welcome to [create an issue on GitHub](https://github.com/FZJ-PGI-12/QAOA.jl/issues). If you would like to contribute to `QAOA.jl`, you can reach us via [PGI-12](https://www.fz-juelich.de/en/pgi/pgi-12).


## Citations

If you are using code from this repository, please [cite our work](https://doi.org/10.21105/joss.05364). Also consider our algorithmic paper:
```
@article{PRXQuantum.4.030335,
  title = {Mean-Field Approximate Optimization Algorithm},
  author = {Misra-Spieldenner, Aditi and Bode, Tim and Schuhmacher, Peter K. and Stollenwerk, Tobias and Bagrets, Dmitry and Wilhelm, Frank K.},
  journal = {PRX Quantum},
  volume = {4},
  issue = {3},
  pages = {030335},
  numpages = {19},
  year = {2023},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/PRXQuantum.4.030335},
  url = {https://link.aps.org/doi/10.1103/PRXQuantum.4.030335}
}
```
