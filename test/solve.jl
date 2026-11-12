@testset "solve finds the optima of small problems" begin
    ring = max_cut(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    r = solve(ring; p = 30, rng = Xoshiro(1))
    @test r.objective == 4
    @test r.x == (1 .+ r.σ) .÷ 2
    @test r.objective ≈ objective(ring, r.σ)
    @test 0 < r.probability ≤ 1
    @test r.schedule == linear_schedule(30)
    @test solve(partition_problem([3.0, 1.0, 1.0, 2.0, 2.0, 1.0]); p = 40, rng = Xoshiro(2)).objective == 0
    cover = solve(min_vertex_cover(4, [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]); p = 30, rng = Xoshiro(3))
    @test cover.objective == 2
    @test cover.x == [1, 0, 1, 0]
    likely = solve(ring, linear_schedule(30); shots = 0)
    @test likely.probability ≈ maximum(probabilities(ring, linear_schedule(30)))
end

"""
    wrong_gradient!(g, x)

The value `sum(x.^2)` with the gradient of its negative: every direction the
optimizer takes leads uphill.
"""
function wrong_gradient!(g, x)
    g .= -2 .* x
    sum(abs2, x)
end

@testset "BFGS safeguards" begin
    # a line search that never succeeds stops at the start
    @test QAOA._bfgs(wrong_gradient!, [1.0, -2.0]; maxiter = 10, gtol = 1e-8) == [1.0, -2.0]
    # an estimate that lost positive definiteness restarts as a scaled identity
    g = [2.0, -1.0]
    H, direction, slope, restarted = QAOA._descent_direction(-Matrix{Float64}(I, 2, 2), g)
    @test restarted && slope < 0
    @test H ≈ 0.05 * I && direction ≈ -0.05 * g
    @test !last(QAOA._descent_direction(Matrix{Float64}(I, 2, 2), g))
end

@testset "optimize_schedule" begin
    prob = max_cut(8, gnp_edges(8, 0.5, 5))
    ws = Workspace(prob)
    start = linear_schedule(2, 0.5)
    sched, value = optimize_schedule(ws, start)
    @test value > expectation(ws, start)
    @test value ≈ expectation(ws, sched)
    _, ∂β, ∂γ = expectation_and_gradient(ws, sched)
    @test maximum(abs, vcat(∂β, ∂γ)) < 1e-4
    values = [last(optimize_schedule(ws, p)) for p in 1:4]
    @test all(diff(values) .> -1e-8)                    # more layers do no worse
    sk = sherrington_kirkpatrick(8)
    _, energy = optimize_schedule(sk, 2)
    @test energy < expectation(sk, linear_schedule(2))  # a minimization goes down
end
