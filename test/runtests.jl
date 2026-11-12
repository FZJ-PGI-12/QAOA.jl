using Test, QAOA, LinearAlgebra, SparseArrays, Random

include("reference.jl")

@testset verbose = true "QAOA.jl" begin
    @testset "Problems" begin
        include("problems.jl")
    end
    @testset "Schedules" begin
        include("schedules.jl")
    end
    @testset "Simulation" begin
        include("simulation.jl")
    end
    @testset "Gradients" begin
        include("gradients.jl")
    end
    @testset "Optimization and solving" begin
        include("solve.jl")
    end
    @testset "Extensions" begin
        include("extensions.jl")
    end
end
