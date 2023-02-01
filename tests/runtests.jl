using Test, PyCall, Yao, Zygote
np = pyimport("numpy")
nx = pyimport("networkx")
include("./../src/QAOA.jl")

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
end