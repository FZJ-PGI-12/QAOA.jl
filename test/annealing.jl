@testset "Toy Annealing" begin
    J = [0.0 0.5428970107716381 -1.3951405027746535 0.13375702345533; 0.5428970107716381 0.0 -0.9460077791129564 0.0036930338557504432; -1.3951405027746535 -0.9460077791129564 0.0 0.5303622417454488; 0.13375702345533 0.0036930338557504432 0.5303622417454488 0.0]
    T_anneal = 128.
    p = 1024
    linear_schedule(t) = t / T_anneal
    annealing_problem = QAOA.Problem(p, zeros(size(J)[1]), J)
    probabilities = anneal(annealing_problem, linear_schedule, T_anneal)

    max_p = maximum(probabilities)
    @test isapprox(max_p, 0.4999, atol=1e-4)
    @test probabilities[findfirst(x -> x == max_p, probabilities)] == probabilities[findlast(x -> x == max_p, probabilities)]
end

@testset "Toy Annealing Second Order" begin
    J = [0.0 0.5428970107716381 -1.3951405027746535 0.13375702345533; 0.5428970107716381 0.0 -0.9460077791129564 0.0036930338557504432; -1.3951405027746535 -0.9460077791129564 0.0 0.5303622417454488; 0.13375702345533 0.0036930338557504432 0.5303622417454488 0.0]
    T_anneal = 128.
    p = 1024
    τ = T_anneal / p
    γ = τ .* ((1:p) .- 1/2) ./ p |> collect
    β = τ .* (1 .- (1:p) ./ p) |> collect
    β[p] = τ / (4 * p)
    annealing_problem = QAOA.Problem(p, zeros(size(J)[1]), J)
    probabilities = anneal(annealing_problem, β, γ)

    max_p = maximum(probabilities)
    @test isapprox(max_p, 0.4999, atol=1e-4)
    @test probabilities[findfirst(x -> x == max_p, probabilities)] == probabilities[findlast(x -> x == max_p, probabilities)]
end