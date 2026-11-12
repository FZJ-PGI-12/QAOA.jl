short = Schedule([0.3, 0.2, 0.1], [0.1, 0.4, 0.5])
cases = [
    ("Ising with fields", Workspace(random_ising(5)), short),
    ("mixer factors", Workspace(random_ising(5; Δ = [0.5, 1.0, 1.5, 0.8, 1.2])), short),
    ("folded MaxCut, paired passes", Workspace(max_cut(14, gnp_edges(14, 0.5, 2))), linear_schedule(3, 0.6)),
    ("full MaxCut, paired passes", Workspace(max_cut(14, gnp_edges(14, 0.5, 2)); symmetry = false), linear_schedule(3, 0.6)),
    ("SK with threads", Workspace(sherrington_kirkpatrick(16); threaded = true), linear_schedule(2, 0.7)),
    ("QUBO as a table", Workspace(min_vertex_cover(6, [(1, 2), (2, 3), (4, 5), (5, 6), (3, 6)])), short),
    ("XY mixer", Workspace(random_ising(5); mixer = XYMixer([(1, 2), (2, 3), (3, 4), (4, 5)]),
                           initial_state = dicke_state(5, 2)), short),
    ("general problem", Workspace(GeneralProblem(x -> sum(x)^3 - 4x[2] * x[5], 5)), short),
]
@testset "$name" for (name, ws, sched) in cases
    E, ∂β, ∂γ = expectation_and_gradient(ws, sched)
    fβ, fγ = finite_difference_gradient(ws, sched)
    size = max(1, maximum(abs, vcat(fβ, fγ)))
    @test E ≈ expectation(ws, sched)
    @test maximum(abs, vcat(∂β - fβ, ∂γ - fγ)) / size < 1e-7
end
@test_throws DimensionMismatch expectation_and_gradient!(zeros(2), zeros(3), Workspace(random_ising(3)), linear_schedule(3))
