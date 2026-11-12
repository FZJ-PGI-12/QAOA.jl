# evolve.jl — running the QAOA and reading off its results.

"""
    evolve!(ws, sched) -> ws

Run the QAOA with the angles `sched` from the initial state. The final state
is then in `ws.state`; [`probabilities`](@ref) and [`statevector`](@ref)
give it in the usual form.
"""
function evolve!(ws::Workspace, sched::Schedule)
    _initialize!(ws)
    for k in 1:length(sched)
        _layer!(ws, Float64(sched.γ[k]), Float64(sched.β[k]))
    end
    ws
end

"""
    expectation(ws, sched) -> Float64
    expectation(prob, sched; kwargs...) -> Float64

The expectation value of the objective in the final state of the QAOA with the
angles `sched`: the average objective of the measured assignments, in the
units of the problem. The second form builds a [`Workspace`](@ref) with the
keywords `kwargs` first.
"""
expectation(ws::Workspace, sched::Schedule) = (evolve!(ws, sched); _objective_mean(ws))
expectation(prob::AbstractProblem, sched::Schedule; kwargs...) =
    expectation(Workspace(prob; kwargs...), sched)

"""
    _objective_mean(ws) -> Float64

The expectation value of the objective in the current state of `ws`.
"""
_objective_mean(ws::Workspace) = ws.sign * ws.scale * _mean_phase_value(ws) + ws.offset

"""
    _mean_phase_value(ws) -> Float64

`⟨f⟩ = Σ_b |ψ_b|² f_b`, the expectation value of the phase per unit angle.
"""
function _mean_phase_value(ws::Workspace)
    ψ, d = ws.state, ws.diagonal
    _chunked_sum(length(ψ), _threaded(ws), 0.0) do lo, hi
        _mean_range(ψ, d, lo, hi)
    end
end

"""
    _mean_range(ψ, d, lo, hi) -> Float64

`Σ_{b ∈ lo:hi} |ψ_b|² f_b`.
"""
function _mean_range(ψ::Vector{ComplexF64}, d, lo::Int, hi::Int)
    m = 0.0
    @inbounds @simd for b in lo:hi
        m += abs2(ψ[b]) * _phase_value(d, b)
    end
    m
end

"""
    probabilities(ws, sched) -> Vector{Float64}
    probabilities(prob, sched; kwargs...) -> Vector{Float64}

The probability of measuring each assignment after the QAOA with the angles
`sched`. Entry `b + 1` belongs to the assignment whose variable `i` is bit
`i - 1` of `b`: for an [`IsingProblem`](@ref), bit 1 means `σ[i] = +1`.
"""
probabilities(ws::Workspace, sched::Schedule) =
    (evolve!(ws, sched); _unfold(abs2.(ws.state) ./ (ws.folded ? 2 : 1), ws.folded))
probabilities(prob::AbstractProblem, sched::Schedule; kwargs...) =
    probabilities(Workspace(prob; kwargs...), sched)

"""
    statevector(ws, sched) -> Vector{ComplexF64}
    statevector(prob, sched; kwargs...) -> Vector{ComplexF64}

The final state of the QAOA with the angles `sched`, as `2^n` amplitudes in
the order of [`probabilities`](@ref).
"""
statevector(ws::Workspace, sched::Schedule) =
    (evolve!(ws, sched); _unfold(ws.state ./ (ws.folded ? sqrt(2) : 1), ws.folded))
statevector(prob::AbstractProblem, sched::Schedule; kwargs...) =
    statevector(Workspace(prob; kwargs...), sched)

"""
    _unfold(v, folded) -> Vector

The values of all `2^n` assignments from those of the stored ones. A folded
state stores the assignments whose last qubit is 0. The complement of the
stored assignment `b - 1` has its last qubit at 1, lies at index
`2N + 1 - b`, and has the same value. The caller converts `v` from the
stored normalization first (probabilities halved, amplitudes divided by √2).
"""
function _unfold(v::Vector{T}, folded::Bool) where {T}
    folded || return v
    N = length(v)
    full = Vector{T}(undef, 2N)
    @inbounds for b in 1:N
        full[b] = v[b]
        full[2N+1-b] = v[b]
    end
    full
end
