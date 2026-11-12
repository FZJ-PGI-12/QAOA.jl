@testset "Validation" begin
    @test_throws ArgumentError IsingProblem(zeros(2), [0 1; 2 0])          # not symmetric
    @test_throws ArgumentError IsingProblem(zeros(2), [1 0; 0 0])          # diagonal entry
    @test_throws ArgumentError IsingProblem(zeros(3), zeros(2, 2))         # size
    @test_throws ArgumentError IsingProblem(zeros(2), zeros(2, 2); sense = :maximize)
    @test_throws ArgumentError IsingProblem(zeros(2), zeros(2, 2); Δ = [1, -1])
    @test_throws ArgumentError QUBOProblem(zeros(2, 3))
    @test_throws ArgumentError GeneralProblem(zeros(3))
end

@testset "Objectives and conversions" begin
    n = 5
    ising = random_ising(n; offset = 1.5)
    J = Matrix(ising.J)
    σ = spins(6, n)
    @test objective(ising, σ) ≈
          dot(ising.h, σ) + sum(J[i, j] * σ[i] * σ[j] for j in 1:n for i in 1:j-1) + 1.5
    qubo = QUBOProblem(randn(Xoshiro(2), n, n); c = randn(Xoshiro(3), n), constant = -2.0)
    as_ising = IsingProblem(qubo)
    @test all(objective(qubo, bits(b, n)) ≈ objective(as_ising, spins(b, n)) for b in 0:2^n-1)
    @test fieldscale(qubo) ≈ fieldscale(as_ising)
    general = GeneralProblem(objectives(ising))
    @test fieldscale(general) ≈ fieldscale(ising)
    @test objective(general, bits(9, n)) ≈ objective(ising, spins(9, n))
    @test fieldscale(IsingProblem(ising.h, sparse(J))) ≈ fieldscale(IsingProblem(ising.h, J))
    @test fieldscale(IsingProblem(ising.h, J; scale = 1)) == 1
    sparse_ising = IsingProblem(QUBOProblem(sparse(qubo.Q); c = qubo.c, constant = -2.0))
    @test sparse_ising.J isa SparseMatrixCSC
    @test Matrix(sparse_ising.J) ≈ as_ising.J
    @test sparse_ising.h ≈ as_ising.h && sparse_ising.offset ≈ as_ising.offset
end

@testset "Display" begin
    @test sprint(show, IsingProblem(zeros(3), zeros(3, 3))) == "IsingProblem{Float64}: 3 spins, sense = :min"
    @test sprint(show, QUBOProblem(zeros(2, 2); sense = :max)) == "QUBOProblem{Float64}: 2 variables, sense = :max"
    @test sprint(show, GeneralProblem(zeros(8))) == "GeneralProblem{Float64}: 3 variables, sense = :min"
    @test startswith(sprint(show, linear_schedule(2)), "Schedule{Float64} with 2 layers: β = ")
    @test startswith(sprint(show, Workspace(max_cut(4, [(1, 2), (2, 3)]))), "Workspace for IsingProblem")
    result = solve(max_cut(4, [(1, 2), (2, 3)]), linear_schedule(2); rng = Xoshiro(1))
    @test startswith(sprint(show, result), "Result{Float64}: 4 variables, objective = ")
end

@testset "Problem builders" begin
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
    weights = [1, 2, 1, 1, 3]
    cut = max_cut(4, edges; weights)
    for b in 0:15
        x = bits(b, 4)
        @test objective(cut, spins(b, 4)) ≈ sum(w for (w, (i, j)) in zip(weights, edges) if x[i] != x[j]; init = 0)
    end
    cover = min_vertex_cover(4, edges)
    @test minimum(objectives(cover)) == 2               # {1, 3} covers all five edges
    a = [3.0, 1.0, 1.0, 2.0, 2.0, 1.0]
    part = partition_problem(a)
    @test all(objective(part, spins(b, 6)) ≈ sum(a .* spins(b, 6))^2 for b in 0:63)
    sk = sherrington_kirkpatrick(6; seed = 4)
    @test sk.J == sherrington_kirkpatrick(6; seed = 4).J
    @test issymmetric(sk.J) && sk.sense === :min
end

@testset "Objective vectors" begin
    problems = (random_ising(6), random_ising(6; sense = :max, offset = 2),
                max_cut(6, gnp_edges(6, 0.6, 1)), min_vertex_cover(6, [(1, 2), (2, 3), (4, 5), (5, 6), (3, 6)]),
                QUBOProblem(randn(Xoshiro(5), 6, 6)), GeneralProblem(x -> sum(x)^3 - 2x[1], 6))
    for prob in problems
        f = phase_vector(prob)
        d = QAOA._diagonal(prob, false)
        @test [QAOA._phase_value(d, b) for b in 1:64] ≈ f atol = 1e-12
        if QAOA._flipsymmetric(prob)
            folded = QAOA._diagonal(prob, true)
            @test [QAOA._phase_value(folded, b) for b in 1:32] ≈ f[1:32] atol = 1e-12
        end
    end
    @test QAOA._diagonal(max_cut(6, gnp_edges(6, 0.6, 1)), false) isa QAOA.TableDiagonal
    @test QAOA._diagonal(random_ising(6), false) isa QAOA.DenseDiagonal
    @test QAOA._diagonal(GeneralProblem(x -> sum(x)^3, 6), false) isa QAOA.TableDiagonal
    # integer weights too large for a table, and values on no grid, fall back to dense
    heavy = IsingProblem(fill(6000.0, 6), zeros(6, 6))
    general = GeneralProblem(x -> π * sum(x), 6)
    for prob in (heavy, general)
        d = QAOA._diagonal(prob, false)
        @test d isa QAOA.DenseDiagonal
        @test [QAOA._phase_value(d, b) for b in 1:64] ≈ phase_vector(prob) atol = 1e-12
    end
end
