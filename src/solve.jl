# solve.jl — measuring the QAOA and reporting the best assignment.

"""
    Result

The outcome of [`solve`](@ref):

- `σ`, `x`: the best measured assignment, as spins -1/+1 and as bits 0/1,
  with `x = (1 + σ) / 2`;
- `objective`: its objective value;
- `expectation`: the expectation value of the objective in the final state;
- `probability`: the probability of measuring exactly this assignment;
- `schedule`: the angles that were used.
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

Run the QAOA with the angles `sched`, measure it `shots` times, and return
the best measured assignment in a [`Result`](@ref). With `shots = 0`, the
most likely assignment is returned instead.

The third form uses [`linear_schedule(p, τ)`](@ref linear_schedule), a
quantum annealing cut into `p` layers. Optimized angles
([`optimize_schedule`](@ref)) usually do better with few layers.
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
