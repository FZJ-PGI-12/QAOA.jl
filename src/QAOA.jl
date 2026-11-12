module QAOA

using Yao, YaoBlocks, Zygote, ChainRulesCore
using LinearAlgebra
using Parameters
using NLopt
using DocStringExtensions

import Distributions, Random

include("problem.jl")
export Problem

include("optimization.jl")
export cost_function, optimize_parameters

include("utils.jl")
export sherrington_kirkpatrick, partition_problem, max_cut, min_vertex_cover

include("circuit.jl")

include("annealing.jl")
export anneal

end