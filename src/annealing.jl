"""
    anneal(problem::Problem, schedule::Function, T::Float64)

An extra function to do quantum annealing for a given problem instead of the QAOA.

### Input 
- `problem::Problem`: A `Problem` instance defining the optimization problem.
- `schedule::Function`: A function specifying the annealing schedule (s. below).
- `T::Float64`: The duration of the annealing run.

### Output
- `probabilities::Vector{Float64}`: The vector of output probabilities over the bitstrings.

### Notes
The function `schedule` should map from ``[0, T]`` to ``[0, 1]`` and have `schedule(0) == 0`, `schedule(T) == 1`. 
In the simplest case of a linear schedule, set `schedule(t) = t/T`. With driver ``\\hat H_D`` and problem Hamiltonian ``\\hat H_P``, the full annealing Hamiltonian then reads

``
\\hat H_A = (1 - t/T) \\hat H_D + (t/T) \\hat H_P
``

"""
function anneal(problem::Problem, schedule::Function, T::Float64)
    @unpack_Problem problem

    @assert schedule(0.) == 0. "Schedule should start at 0."
    @assert schedule(T) == 1. "Schedule finish stop at 1."

    τ = T/(num_layers - 1)  
    γ = map(schedule, τ .* (0:num_layers-1))
    β = 1 .- γ
    beta_and_gamma = τ .* vcat(β, γ)
    circ = circuit(problem)
    circ = dispatch_parameters!(circ, problem, beta_and_gamma)
    probabilities = uniform_state(nqubits(circ)) |> circ |> probs
    probabilities
end


"""
    anneal(problem::Problem, beta::Vector{Float64}, gamma::Vector{Float64})

An extra function to do quantum annealing for a given problem instead of the QAOA.

### Input 
- `problem::Problem`: A `Problem` instance defining the optimization problem.
- `beta::Vector{Float64}`: A vector of schedule parameters multiplying the transverse-field or driver Hamiltonian.
- `gamma::Vector{Float64}`: A vector of schedule parameters multiplying the problem or computational-basis Hamiltonian.

### Output
- `probabilities::Vector{Float64}`: The vector of output probabilities over the bitstrings.

### Notes
A good choice for the parameters `beta` and `gamma` is given in Appendix A of https://arxiv.org/abs/1907.02359.
``

"""
function anneal(problem::Problem, beta::Vector{Float64}, gamma::Vector{Float64})
    @unpack_Problem problem

    beta_and_gamma = vcat(beta, gamma)
    circ = circuit(problem)
    circ = dispatch_parameters!(circ, problem, beta_and_gamma)
    probabilities = uniform_state(nqubits(circ)) |> circ |> probs
    probabilities
end