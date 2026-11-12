# gradient.jl — derivatives of the expectation value with respect to the angles.
#
# The adjoint method: after the forward run, λ = f ψ; going back through the layers,
# both vectors are undone together, and each layer's derivatives are inner products
# of λ and ψ at that layer:
#   ∂⟨f⟩/∂γ[k] = 2 Im⟨λ|f|ψ⟩,   ∂⟨f⟩/∂β[k] = 2 Σᵢ Δ[i] Im⟨λ|Xᵢ|ψ⟩.
# This costs about two more runs of the QAOA, whatever the number of layers.

"""
    expectation_and_gradient!(∂β, ∂γ, ws, sched) -> Float64

Returns the [`expectation`](@ref) value of the cost function after the QAOA with the angles `sched`, and writes its derivatives with respect to `sched.β` and `sched.γ` into `∂β` and `∂γ`.

### Notes
The derivatives come from one backward pass through the layers (the adjoint method). This costs about as much as two more runs of the QAOA, for any number of layers, and needs one more state vector, which the workspace allocates the first time.
"""
function expectation_and_gradient!(∂β::AbstractVector, ∂γ::AbstractVector, ws::Workspace, sched::Schedule)
    p = length(sched)
    length(∂β) == length(∂γ) == p ||
        throw(DimensionMismatch("∂β and ∂γ need one entry per layer, $p"))
    evolve!(ws, sched)
    mean_f = _seed_adjoint!(ws)
    for k in p:-1:1
        ∂β[k], ∂γ[k] = _layer_adjoint!(ws, Float64(sched.γ[k]), Float64(sched.β[k]))
    end
    factor = ws.sign * ws.scale
    ∂β .*= 2factor
    ∂γ .*= 2factor
    factor * mean_f + ws.offset
end

"""
    expectation_and_gradient(ws, sched) -> (E, ∂β, ∂γ)
    expectation_and_gradient(prob, sched; kwargs...) -> (E, ∂β, ∂γ)

Returns the expectation value `E` of the cost function and its derivatives with respect to the angles `β` and `γ`, see [`expectation_and_gradient!`](@ref). Given a problem `prob`, a [`Workspace`](@ref) is built first, with the keywords `kwargs`.
"""
function expectation_and_gradient(ws::Workspace, sched::Schedule)
    ∂β = zeros(length(sched))
    ∂γ = zeros(length(sched))
    E = expectation_and_gradient!(∂β, ∂γ, ws, sched)
    E, ∂β, ∂γ
end
expectation_and_gradient(prob::AbstractProblem, sched::Schedule; kwargs...) =
    expectation_and_gradient(Workspace(prob; kwargs...), sched)

"""
    _seed_adjoint!(ws) -> ⟨f⟩

Set the adjoint vector to `λ = f ψ`, allocating it the first time, and return
`⟨f⟩ = ⟨ψ|f|ψ⟩` of the current state.
"""
function _seed_adjoint!(ws::Workspace)
    length(ws.adjoint) == length(ws.state) || (ws.adjoint = similar(ws.state))
    ψ, λ, d = ws.state, ws.adjoint, ws.diagonal
    _chunked_sum(length(ψ), _threaded(ws), 0.0) do lo, hi
        _seed_range!(λ, ψ, d, lo, hi)
    end
end

"""
    _seed_range!(λ, ψ, d, lo, hi) -> Σ f|ψ|²

`λ[b] = f_b ψ[b]` for `b ∈ lo:hi`, returning this range's part of `⟨f⟩`.
"""
function _seed_range!(λ::Vector{ComplexF64}, ψ::Vector{ComplexF64}, d, lo::Int, hi::Int)
    m = 0.0
    @inbounds @simd for b in lo:hi
        fb = _phase_value(d, b)
        λ[b] = fb * ψ[b]
        m += fb * abs2(ψ[b])
    end
    m
end
