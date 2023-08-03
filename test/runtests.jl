using Test, QAOA, Yao, Zygote, LinearAlgebra

@testset verbose=true "QAOA.jl" begin

    @testset "QAOA" begin
        include("QAOA.jl")
    end

    @testset "Mean-Field" begin
      include("mean_field.jl")
    end
  
    @testset "Fluctuations" begin
      include("fluctuations.jl")
    end

    @testset "Annealing" begin
      include("annealing.jl")
  end
end