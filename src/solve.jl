# solve.jl — measuring the QAOA and reporting the best assignment.

"""
    Result

A container holding the outcome of [`solve`](@ref).

### Fields
- `σ`, `x`: The best measured bitstring, as spins -1/+1 and as bits 0/1, with ``x_i = (1 + \\sigma_i)/2``.
- `objective`: Its cost function.
- `expectation`: The expectation value of the cost function in the final state.
- `probability`: The probability of measuring exactly this bitstring.
- `schedule`: The angles that were used.
"""
struct Result{T<:AbstractFloat}
    σ::Vector{Int8}
    x::Vector{Int8}
    objective::T
    expectation::T
    probability::T
    schedule::Schedule{T}
end

"""
    show(io, r::Result)

Print the size of the assignment, its objective and the expectation value.
"""
function Base.show(io::IO, r::Result{T}) where {T}
    print(io, "Result{$T}: $(length(r.σ)) variables, objective = $(r.objective), ",
          "expectation = $(r.expectation)")
end

"""
    solve(ws, sched; shots = 1000, rng = Random.default_rng()) -> Result
    solve(prob, sched; shots, rng, kwargs...) -> Result
    solve(prob; p = 20, τ = 0.5, shots, rng, kwargs...) -> Result

Runs the QAOA with the angles `sched`, measures the final state `shots` times, and returns the best measured bitstring.

### Input
- `ws::Workspace`: The workspace of the problem. Given a problem `prob` instead, a [`Workspace`](@ref) is built first, with the keywords `kwargs`.
- `sched::Schedule`: The angles. Without `sched`, [`linear_schedule(p, τ)`](@ref linear_schedule) is used, i.e. a quantum annealing cut into `p` layers.
- `shots::Integer=1000` (optional): The number of measurements. With `shots = 0`, the most likely bitstring is returned instead.
- `rng` (optional): The random-number generator for the measurements.

### Output
- A [`Result`](@ref).

### Notes
With few layers, optimized angles from [`optimize_schedule`](@ref) usually do better than the linear schedule.
"""
function solve(ws::Workspace, sched::Schedule; shots::Integer = 1000, rng::AbstractRNG = Random.default_rng())
    shots ≥ 0 || throw(ArgumentError("shots must be nonnegative, got $shots"))
    evolve!(ws, sched)
    probability = abs2.(ws.state)
    b = shots == 0 ? argmax(probability) : _best_sample(ws, probability, shots, rng)
    x = Int8[(b - 1) >> (i - 1) & 1 for i in 1:nvariables(ws.problem)]
    σ = Int8(2) .* x .- Int8(1)
    T = eltype(ws.problem)
    Result{T}(σ, x, T(_objective_of(ws.problem, σ, x)), T(_objective_mean(ws)),
              T(probability[b] / (ws.folded ? 2 : 1)), Schedule{T}(T.(sched.β), T.(sched.γ)))
end
solve(prob::AbstractProblem, sched::Schedule; shots::Integer = 1000,
      rng::AbstractRNG = Random.default_rng(), kwargs...) =
    solve(Workspace(prob; kwargs...), sched; shots, rng)
solve(prob::AbstractProblem; p::Integer = 20, τ::Real = 0.5, shots::Integer = 1000,
      rng::AbstractRNG = Random.default_rng(), kwargs...) =
    solve(prob, linear_schedule(p, τ); shots, rng, kwargs...)

"""
    _objective_of(prob, σ, x)

The objective of an assignment given both as spins `σ` and bits `x`: each
problem type reads the form it is written in.
"""
_objective_of(prob::IsingProblem, σ, x) = objective(prob, σ)
_objective_of(prob::Union{QUBOProblem,GeneralProblem}, σ, x) = objective(prob, x)

"""
    _best_sample(ws, probability, shots, rng) -> b

Draw `shots` assignments with the given probabilities (of the stored
assignments) and return the index of the best one. The uniform random numbers
are sorted, so one walk through the cumulative probabilities finds all
samples.
"""
function _best_sample(ws::Workspace, probability::Vector{Float64}, shots::Integer, rng::AbstractRNG)
    draws = sort!(rand(rng, shots) .* sum(probability))
    best, best_value = 1, -Inf
    cumulative = 0.0
    b = 0
    for u in draws
        while cumulative < u && b < length(probability)
            b += 1
            cumulative += probability[b]
        end
        value = _phase_value(ws.diagonal, max(b, 1))   # larger is better for every sense
        if value > best_value
            best, best_value = max(b, 1), value
        end
    end
    best
end
