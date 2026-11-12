# Welcome!

`QAOA.jl` is a lightweight implementation of the [Quantum Approximate Optimization Algorithm (QAOA)](https://arxiv.org/abs/1411.4028) based on [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl).

!!! note "The mean-field AOA has moved to MFAOA.jl"
    Up to version 1.4, `QAOA.jl` also implemented the [Mean-Field Approximate Optimization Algorithm](https://doi.org/10.1103/PRXQuantum.4.030335) (mean-field AOA). That algorithm now has its own package, [`MFAOA.jl`](https://github.com/Quicopt/MFAOA.jl), which also accepts QUBO problems and works as a JuMP optimizer. The Gaussian fluctuation analysis of the mean-field AOA (`evolve_fluctuations`) is available in `QAOA.jl` 1.4 and earlier.

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```


## Library

### Index

!!! note
    Below is a list of the functions exported by `QAOA.jl`. Please note that we are taking advantage of [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) as implemented in `Julia`, i.e. some functions are defined multiple times with different signatures.

```@index
```

### Problem Structure

```@docs
Problem{}
```

### Cost Function and QAOA Parameter Optimization

```@docs
cost_function(problem::Problem, beta_and_gamma::Vector{Float64})
optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}, algorithm; niter::Int=128)
optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}; niter::Int=128, learning_rate::Float64 = 0.05)
```

### Predefined Optimization Problems

```@docs
sherrington_kirkpatrick(N::Int, variance::Float64; seed::Int=1, num_layers::Int=1, driver=X)
partition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)
max_cut(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)
min_vertex_cover(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)
```

### Annealing

```@docs
anneal(problem::Problem, schedule::Function, T_anneal::Float64)
```