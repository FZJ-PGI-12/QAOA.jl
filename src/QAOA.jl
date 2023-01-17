module QAOA

using Yao, YaoBlocks, Zygote
using Parameters
using NLopt
using PyCall
using DocStringExtensions

include("problem.jl")
export Problem

include("circuit.jl")

include("optimization.jl")

include("utils.jl")
export sherrington_kirkpatrick, partition_problem, max_cut, min_vertex_cover

end