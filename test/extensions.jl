import Graphs

@testset "Graphs.jl" begin
    ring = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    g = Graphs.SimpleGraph(5)
    for (i, j) in ring
        Graphs.add_edge!(g, i, j)
    end
    @test max_cut(g).J == max_cut(5, ring).J
    @test max_cut(g).offset == max_cut(5, ring).offset
    @test min_vertex_cover(g).Q == min_vertex_cover(5, ring).Q
end

if Sys.WORD_SIZE == 64
    import Yao
    @testset "Yao.jl circuits" begin
        sched = Schedule([0.3, 0.2], [0.4, 0.1])
        prob = random_ising(5; Δ = [0.5, 1.0, 1.5, 0.8, 1.2], sense = :max)
        state(circuit, ψ0) = Yao.statevec(Yao.apply(Yao.ArrayReg(ComplexF64.(ψ0)), circuit))
        zero = [b == 0 ? 1.0 : 0.0 for b in 0:31]
        @test state(QAOA.circuit(prob, sched), zero) ≈ statevector(prob, sched)
        pairs = [(1, 2), (2, 3), (3, 4), (4, 5)]
        ψ0 = dicke_state(5, 2)
        xy = XYMixer(pairs)
        @test state(QAOA.circuit(prob, sched; mixer = xy), ψ0) ≈
              statevector(prob, sched; mixer = xy, initial_state = ψ0)
        cover = min_vertex_cover(5, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        @test abs(dot(state(QAOA.circuit(cover, sched), zero), statevector(cover, sched))) ≈ 1
        @test_throws ArgumentError QAOA.circuit(GeneralProblem(zeros(4)), sched)
    end
end
