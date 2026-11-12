# QAOA.jl

`QAOA.jl` is a lightweight and efficient implementation of the [Quantum Approximate Optimization Algorithm (QAOA)](https://arxiv.org/abs/1411.4028), built to support the extensive classical simulations typically required in research on the topic. `QAOA.jl` keeps the quantum state of all qubits in memory and thus gives exact probabilities and expectation values, together with their gradients with respect to the angles of the algorithm. One expectation value for 20 qubits takes a few milliseconds, and 30 qubits fit into the memory of a laptop. A number of common optimization problems such as MaxCut, the minimum vertex-cover problem, the Sherrington-Kirkpatrick model, and the partition problem are pre-implemented to facilitate scientific benchmarking.

!!! note "The mean-field AOA has moved to MFAOA.jl"
    Up to version 1.4, `QAOA.jl` also implemented the [Mean-Field Approximate Optimization Algorithm](https://doi.org/10.1103/PRXQuantum.4.030335) (mean-field AOA). That algorithm now has its own package, [`MFAOA.jl`](https://github.com/Quicopt/MFAOA.jl). Both packages describe a problem and its angles in the same way, so we can run the QAOA with `QAOA.jl` and its mean-field approximation with `MFAOA.jl` in the same script. Note that the two packages export many of the same names; load the second one with `import MFAOA` and write `MFAOA.solve` and so on. The Gaussian fluctuation analysis of the mean-field AOA (`evolve_fluctuations`) is available in `QAOA.jl` 1.4 and earlier.

## Installation

To install, use Julia's built-in package manager

```julia
julia> ] add QAOA
```

## A First Example

The MaxCut problem splits the nodes of a graph into two groups such that as many edges as possible join nodes of different groups. A ring of five nodes has five edges, and at most four of them can be cut. We can solve this problem with `QAOA.jl` as follows:

```@example first
using QAOA, Random
ring = max_cut(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
result = solve(ring; p = 20, rng = Xoshiro(1))
```

[`solve`](@ref) runs the QAOA with 20 layers, measures the final quantum state 1000 times and returns the best bitstring it saw. Its objective is the number of edges cut, and the bits `x` give the group of each node:

```@example first
result.objective, result.x
```

The expectation value, i.e. the average cut over all measurements, comes close to the maximum:

```@example first
result.expectation
```

## The Algorithm

For a problem with ``n`` binary variables, the QAOA uses ``n`` qubits. When we measure them, qubit ``i`` in the state ``|1\rangle`` means ``x_i = 1``, or the spin ``\sigma_i = +1`` in an Ising problem. The QAOA prepares the state

```math
\begin{align*}
|\beta, \gamma\rangle = e^{-i \beta_p \hat B} e^{-i \gamma_p \hat C / s} \cdots e^{-i \beta_1 \hat B} e^{-i \gamma_1 \hat C / s} |+\rangle^{\otimes n},
\end{align*}
```

where ``\hat C`` is the cost function as an operator, ``\hat C |x\rangle = C(x) |x\rangle``, and ``\hat B = \sum_i \hat X_i`` is the mixer. The state ``|+\rangle^{\otimes n}`` gives all ``2^n`` bitstrings the same amplitude. Each of the ``p`` layers then does two things:

1. The _phase separator_ ``e^{-i \gamma_k \hat C / s}`` multiplies the amplitude of every bitstring by a phase that depends on its cost function. For a minimization, the phase has the opposite sign.
2. The _mixer_ ``e^{-i \beta_k \hat B}`` turns every qubit, which moves amplitude between bitstrings that differ in one bit.

The angles ``\beta_k`` and ``\gamma_k`` form a [`Schedule`](@ref). With good angles, the bitstrings with good cost functions end up with large amplitudes, so a measurement is likely to return one of them.

The scale ``s`` is the [`fieldscale`](@ref) of the problem. Since we measure ``\gamma`` in units of ``s``, good angles do not depend on the units of the cost function or on the size of the problem. With many layers and slowly changing angles, the QAOA becomes a quantum annealing from the mixer to the cost function, and [`linear_schedule`](@ref) builds such a schedule. With few layers, optimized angles do better; [`optimize_schedule`](@ref) finds them.

## Problems

`QAOA.jl` describes a problem by one of three types:

- [`IsingProblem`](@ref) has spins ``\sigma_i = \pm 1`` and the cost function ``C(\sigma) = \sum_i h_i \sigma_i + \sum_{i<j} J_{ij} \sigma_i \sigma_j + \mathrm{offset}``.
- [`QUBOProblem`](@ref) has bits ``x_i \in \{0, 1\}`` and the cost function ``C(x) = \sum_{i,j} Q_{ij} x_i x_j + \sum_i c_i x_i + \mathrm{constant}``.
- [`GeneralProblem`](@ref) takes any cost function, as a function of the bits or as a table of its values.

Each problem is _minimized_ by default; `sense = :max` maximizes it. The wrapper functions [`max_cut`](@ref), [`min_vertex_cover`](@ref), [`sherrington_kirkpatrick`](@ref) and [`partition_problem`](@ref) set up common problems. With [`Graphs.jl`](https://github.com/JuliaGraphs/Graphs.jl) loaded, `max_cut` and `min_vertex_cover` also take a graph.

## Workspaces

A [`Workspace`](@ref) holds everything the QAOA needs for one problem: the cost function of every bitstring and the quantum state. [`expectation`](@ref), [`probabilities`](@ref), [`statevector`](@ref), [`expectation_and_gradient`](@ref) and [`solve`](@ref) take a workspace or a problem. Given a problem, they build a workspace first. When we evaluate many schedules for the same problem, as in an optimization of the angles, we should reuse one workspace:

```@example first
ws = Workspace(ring)
tuned, value = optimize_schedule(ws, 3)
value
```

Our page on [performance](@ref Performance) explains how the workspace saves time and memory, and how to use several threads.

## Library

### Index

!!! note
    Below is a list of the functions exported by `QAOA.jl`. Please note that we are taking advantage of [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) as implemented in `Julia`, i.e. some functions are defined multiple times with different signatures.

```@index
```

### Problems

```@docs
IsingProblem
QUBOProblem
GeneralProblem
objective
fieldscale
QAOA.nvariables
```

### Predefined Optimization Problems

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

### Mixers and Initial States

```@docs
XMixer
XYMixer
dicke_state
```

### Circuits

```@docs
QAOA.circuit
```
