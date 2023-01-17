# Welcome!

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```


## Library

### Index

```@index
```


### Predefined Optimization Problems

```@docs
sherrington_kirkpatrick(N::Int, variance::Float64; seed::Int=1, num_layers::Int=1, driver=X)
partition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)
max_cut(graph::PyObject; num_layers::Int=1, driver=X)
min_vertex_cover(graph::PyObject; num_layers::Int=1, driver=X)
```
