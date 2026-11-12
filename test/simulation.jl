@testset "Exact states for small n" begin
    sched = Schedule([0.3, 0.2, 0.1], [0.1, 0.4, 0.5])
    for n in 1:6
        prob = random_ising(n; seed = n)
        @test statevector(prob, sched) ≈ exact_state(prob, sched) atol = 1e-12
    end
    weighted = random_ising(5; Δ = [0.5, 1.0, 1.5, 0.8, 1.2])
    @test statevector(weighted, sched) ≈ exact_state(weighted, sched; Δ = weighted.Δ) atol = 1e-12
    general = GeneralProblem(x -> sum(x)^3 - 4x[2] * x[5], 5; sense = :max)
    @test statevector(general, sched) ≈ exact_state(general, sched) atol = 1e-12
    cover = min_vertex_cover(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    @test statevector(cover, sched) ≈ exact_state(cover, sched) atol = 1e-12
end

@testset "XY mixer and initial states" begin
    sched = Schedule([0.3, 0.2, 0.1], [0.1, 0.4, 0.5])
    prob = random_ising(5)
    pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    ψ0 = dicke_state(5, 2)
    ψ = statevector(prob, sched; mixer = XYMixer(pairs), initial_state = ψ0)
    @test ψ ≈ exact_state(prob, sched; pairs, ψ0) atol = 1e-12
    @test sum(abs2(ψ[b+1]) for b in 0:31 if count_ones(b) == 2) ≈ 1      # weight 2 is kept
    ψr = randn(Xoshiro(9), ComplexF64, 32)
    @test statevector(prob, sched; initial_state = ψr) ≈ exact_state(prob, sched; ψ0 = ψr) atol = 1e-12
    @test_throws ArgumentError Workspace(prob; mixer = XYMixer([(1, 6)]))
    @test_throws ArgumentError XYMixer([(2, 2)])
end

@testset "Blocked and paired passes, folding, threads" begin
    # beyond 11 qubits, the high qubits are turned two per pass: odd and even counts
    sched = Schedule([0.5, 0.3], [0.2, 0.6])
    for n in 12:16
        for prob in (random_ising(n; seed = n), max_cut(n, gnp_edges(n, 0.5, n)))
            reference = loop_state(prob, sched)
            for symmetry in (true, false), threaded in (false, true)
                ws = Workspace(prob; symmetry, threaded)
                @test statevector(ws, sched) ≈ reference atol = 1e-12
            end
        end
    end
end

@testset "Reference value of QAOA.jl 1.4 and JuliQAOA" begin
    n = 10
    cut = max_cut(n, gnp_edges(n, 0.5, n))
    raw = IsingProblem(cut.h, cut.J; offset = cut.offset, sense = :max, scale = 1)
    @test expectation(raw, Schedule([0.4, 0.37, 0.34], [0.2, 0.25, 0.3])) ≈ 13.6029308535638 atol = 1e-12
end

@testset "Probabilities and expectation values" begin
    prob = max_cut(9, gnp_edges(9, 0.5, 3))
    sched = linear_schedule(5)
    folded = probabilities(prob, sched)
    @test sum(folded) ≈ 1
    @test folded ≈ probabilities(prob, sched; symmetry = false) atol = 1e-14
    @test folded ≈ abs2.(statevector(prob, sched))
    @test expectation(prob, sched) ≈ sum(folded .* objectives(prob))
    @test Workspace(prob).folded && !Workspace(prob; symmetry = false).folded
end
