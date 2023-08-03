function anneal(problem::Problem, schedule::Function, T_anneal::Float64)
    @unpack_Problem problem

    @assert schedule(0.) == 0. "Schedule should start at 0."
    @assert schedule(T_anneal) == 1. "Schedule finish stop at 1."

    τ = T_anneal/(num_layers - 1)  
    γ = map(schedule, τ .* (0:num_layers-1))
    β = 1 .- γ
    beta_and_gamma = vcat(β, γ)
    circ = circuit(problem)
    circ = dispatch_parameters!(circ, problem, beta_and_gamma)
    probabilities = uniform_state(nqubits(circ)) |> circ |> probs
    probabilities
end