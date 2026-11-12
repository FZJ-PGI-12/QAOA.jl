# QAOA.jl

`QAOA.jl` simulates the [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028)
(QAOA) on a classical computer. The QAOA searches for good solutions of
optimization problems over binary variables, such as MaxCut. The package
keeps the quantum state of all qubits in memory. It therefore gives exact
probabilities and expectation values, together with their derivatives with
respect to the angles of the algorithm. One evaluation for 20 qubits takes a
few milliseconds, and 30 qubits fit into the memory of a laptop.

!!! note "The mean-field AOA has moved to MFAOA.jl"
    Up to version 1.4, `QAOA.jl` also implemented the [Mean-Field Approximate Optimization Algorithm](https://doi.org/10.1103/PRXQuantum.4.030335)
    (mean-field AOA). That algorithm now has its own package, [`MFAOA.jl`](https://github.com/Quicopt/MFAOA.jl).
    Both packages describe a problem and its angles in the same way, so one
    script can run the QAOA with `QAOA.jl` and its mean-field approximation
    with `MFAOA.jl`. The two packages export many of the same names; load
    the second with `import MFAOA` and write `MFAOA.solve` and so on. The
    Gaussian fluctuation analysis of the mean-field AOA (`evolve_fluctuations`)
    is available in `QAOA.jl` 1.4 and earlier.

## Installation

```julia
julia> ] add QAOA
```

## A first example

MaxCut splits the nodes of a graph into two groups so that as many edges as
possible join nodes of different groups. A ring of five nodes has five
edges, and at most four of them can join different groups:

```@example first
using QAOA, Random
ring = max_cut(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
result = solve(ring; p = 20, rng = Xoshiro(1))
```

[`solve`](@ref) runs the QAOA with 20 layers, measures the final quantum state
1000 times and returns the best measured assignment. Its objective is the
number of edges cut, and the bits `x` give the group of each node:

```@example first
result.objective, result.x
```

The expectation value is the average cut over all measurements. It comes
close to the maximum:

```@example first
result.expectation
```

## The algorithm

For a problem with `n` binary variables, the QAOA uses `n` qubits. Measuring
qubit `i` as 1 means `x[i] = 1`, or the spin `σ[i] = +1` in an Ising problem.
The algorithm starts with every qubit in the state |+⟩, which gives all `2^n`
assignments the same amplitude. Each of its `p` layers then does two things:

1. The *phase separator* multiplies the amplitude of every assignment by the
   phase `exp(-iγ C / s)`, with the objective `C` of that assignment and a
   scale `s` of the problem, explained below. For a minimization, the phase
   has the opposite sign.
2. The *mixer* turns every qubit by `exp(-iβ X)`, which moves amplitude
   between assignments that differ in one bit.

The angles `β[k]` and `γ[k]` of the layers form a [`Schedule`](@ref). With
good angles, the assignments with good objectives end up with large
amplitudes, so a measurement is likely to return one of them.

The scale `s` is the [`fieldscale`](@ref) of the problem. Measuring `γ` in
units of `s` makes good angles independent of the units of the objective and
of the size of the problem. With many layers and slowly changing angles, the
QAOA becomes a quantum annealing from the mixer to the objective, and
[`linear_schedule`](@ref) builds such a schedule. With few layers, optimized
angles do better; [`optimize_schedule`](@ref) finds them.

## Problems

Three types describe a problem:

- [`IsingProblem`](@ref) has spins `σ[i] = ±1` and the objective
  `h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j] + offset`.
- [`QUBOProblem`](@ref) has bits `x[i] ∈ {0, 1}` and the objective
  `x' * Q * x + c' * x + constant`.
- [`GeneralProblem`](@ref) takes any objective, as a function of the bits or
  as a table of its values.

Each problem is minimized by default; `sense = :max` maximizes it.
[`max_cut`](@ref), [`min_vertex_cover`](@ref),
[`sherrington_kirkpatrick`](@ref) and [`partition_problem`](@ref) build
common problems. With [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)
loaded, `max_cut` and `min_vertex_cover` also take a graph.

## Workspaces

A [`Workspace`](@ref) holds what the QAOA needs for one problem: the objective
of every assignment and the quantum state. [`expectation`](@ref),
[`probabilities`](@ref), [`statevector`](@ref),
[`expectation_and_gradient`](@ref) and [`solve`](@ref) take a workspace or a
problem. Given a problem, they build a workspace first. A search over angles
evaluates many schedules for the same problem, so it should reuse one
workspace:

```@example first
ws = Workspace(ring)
tuned, value = optimize_schedule(ws, 3)
value
```

The page on [performance](@ref Performance) explains how the workspace
saves time and memory, and how to use several threads.

## Library

### Problems

```@docs
IsingProblem
QUBOProblem
GeneralProblem
objective
fieldscale
QAOA.nvariables
```

### Problem builders

```@docs
max_cut
min_vertex_cover
sherrington_kirkpatrick
partition_problem
```

### Schedules

```@docs
Schedule
linear_schedule
interpolate_schedule
optimize_schedule
```

### Running the QAOA

```@docs
Workspace
evolve!
expectation
expectation_and_gradient
expectation_and_gradient!
probabilities
statevector
solve
Result
```

### Mixers and initial states

```@docs
XMixer
XYMixer
dicke_state
```

### Circuits

```@docs
QAOA.circuit
```
