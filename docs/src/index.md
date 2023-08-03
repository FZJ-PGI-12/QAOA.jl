# Welcome!

`QAOA.jl` is a lightweight implementation of the [Quantum Approximate Optimization Algorithm (QAOA)](https://arxiv.org/abs/1411.4028) based on [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl). Furthermore, it implements the [mean-field Approximate Optimization Algorithm](https://arxiv.org/abs/2303.00329), which is a useful tool to simulate quantum annealing and the QAOA in the mean-field approximation.

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

### Mean-Field Approximate Optimization Algorithm

```@docs
evolve!(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
evolve!(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
expectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
mean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})
mean_field_solution(S::Vector{<:Vector{<:Real}})
```

### Fluctuation Analysis

```@docs
evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})
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