using Test, PyCall
np = pyimport("numpy")
include("./../src/QAOA.jl")

@testset "SK model" begin

    # schedule
    p = 10
    τ = 0.5
    γ = τ * (np.arange(1, p + 1) .- 1/2) / p
    β = τ * (1 .- np.arange(1, p + 1) / p)
    β[p] = τ / (4 * p)


    # initial spins
    N = 5
    S = [[1., 0., 0.] for _ in 1:N-1] # fix final spin (i.e. leave it out)

    # SK model
    np.random.seed(11)
    J = np.random.normal(0, 1, size=(N, N)) / np.sqrt(N)
    J = np.triu(J, k=1)
    J = J + transpose(J)

    # evolution
    S = QAOA.evolve(S, J, β, γ)
    
    # solution
    S_test = [[-0.4280189887648497,   0.57845514021309,     0.6943985858408479],
              [-0.4161436735954462,   0.1965348184395327,  -0.8878054449413042],
              [-0.3151885316091266,  -0.14809317283731896, -0.9374031159010826],
              [ 0.06825210700086091, -0.5908423629410464,  -0.8038948638001017]]
    
    @test S ≈ S_test rtol = 1e-10                       
    # @assert np.allclose(expectation(S[:, 2], J), -2.5367551470078142)
    # @assert np.allclose(solution(S[:, 2]), [ 1., -1., -1., -1.])
end