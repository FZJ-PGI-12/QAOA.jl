"""
    QAOA

Simulation of the Quantum Approximate Optimization Algorithm (QAOA) on a
classical computer. The state of all qubits is kept in memory, and every
layer of the algorithm is applied to it directly, without building a circuit.
"""
module QAOA

using LinearAlgebra
using Random
using SparseArrays
using Base.Threads: @threads, nthreads

export IsingProblem, QUBOProblem, GeneralProblem, objective, fieldscale
export max_cut, min_vertex_cover, sherrington_kirkpatrick, partition_problem
export Schedule, linear_schedule, interpolate_schedule
export XMixer, XYMixer, dicke_state
export Workspace, evolve!, expectation, expectation_and_gradient, expectation_and_gradient!
export probabilities, statevector
export optimize_schedule, Result, solve

include("problems.jl")
include("instances.jl")
include("schedule.jl")
include("diagonal.jl")
include("kernels.jl")
include("workspace.jl")
include("layers.jl")
include("evolve.jl")
include("gradient.jl")
include("optimize.jl")
include("solve.jl")

"""
    QAOA.circuit(prob, sched)

Returns the QAOA for `prob` with the angles `sched` as a [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl) circuit, to run it elsewhere or to export it. Note that this function is available only after `using Yao`.
"""
function circuit end

end
