module QAOA

using Yao, YaoBlocks, Zygote
using LinearAlgebra
using Parameters
using NLopt
# using PyCall
using DocStringExtensions

import Distributions, Random

include("problem.jl")
export Problem

include("optimization.jl")
export cost_function, optimize_parameters

include("utils.jl")
export sherrington_kirkpatrick, partition_problem, max_cut, min_vertex_cover

include("circuit.jl")

include("mean_field.jl")
export evolve, expectation, mean_field_solution

include("fluctuations.jl")
export evolve_fluctuations

end