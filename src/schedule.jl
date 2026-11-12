# schedule.jl — the angles of the QAOA layers.

"""
    Schedule(β, γ)

The angles of a QAOA with `p` layers, one pair `β[k]`, `γ[k]` per layer `k`.

The QAOA starts with every qubit in state |+⟩, which gives every assignment
the same amplitude. Each layer `k` then does two things:

1. It multiplies the amplitude of every assignment by the phase
   `exp(-iγ[k] C / s)`, where `C` is the objective of that assignment and `s`
   the problem's [`fieldscale`](@ref). For a minimization, the phase is
   `exp(+iγ[k] C / s)`.
2. It applies the mixer with angle `β[k]`. The default mixer turns every
   qubit `i` by `exp(-iβ[k] Δ[i] X[i])`, where `Δ[i]` is 1 unless the problem
   sets it.

`β` and `γ` must have the same length `p`. [`linear_schedule`](@ref) builds the
usual starting point, and [`optimize_schedule`](@ref) tunes the angles.
"""
struct Schedule{T<:AbstractFloat}
    β::Vector{T}
    γ::Vector{T}
    function Schedule{T}(β::Vector{T}, γ::Vector{T}) where {T<:AbstractFloat}
        length(β) == length(γ) || throw(ArgumentError(
            "β and γ must have one entry per layer; got $(length(β)) and $(length(γ))"))
        isempty(β) && throw(ArgumentError("a schedule needs at least one layer"))
        new{T}(β, γ)
    end
end

"""
    Schedule(β::AbstractVector, γ::AbstractVector)

Build a schedule from any two real vectors, promoted to a common
floating-point type.
"""
function Schedule(β::AbstractVector{<:Real}, γ::AbstractVector{<:Real})
    T = float(promote_type(eltype(β), eltype(γ)))
    Schedule{T}(Vector{T}(β), Vector{T}(γ))
end

"""
    length(sched::Schedule)

The number of layers `p`.
"""
Base.length(sched::Schedule) = length(sched.β)

"""
    sched1 == sched2

Whether two schedules have the same angles.
"""
Base.:(==)(a::Schedule, b::Schedule) = a.β == b.β && a.γ == b.γ

"""
    isapprox(sched1, sched2; kwargs...)

Whether two schedules have approximately the same angles, with the keywords of
`isapprox` for vectors.
"""
Base.isapprox(a::Schedule, b::Schedule; kwargs...) =
    length(a) == length(b) && isapprox(a.β, b.β; kwargs...) && isapprox(a.γ, b.γ; kwargs...)

"""
    show(io, sched::Schedule)

Print the number of layers and both angle vectors.
"""
function Base.show(io::IO, sched::Schedule{T}) where {T}
    print(io, "Schedule{$T} with $(length(sched)) layers: β = ", sched.β, ", γ = ", sched.γ)
end

"""
    linear_schedule(p, τ = 0.5; T = Float64)

The usual starting schedule: over `p` layers, the angles `γ` grow and the
angles `β` shrink linearly,

    γ[k] = τ k / p,    β[k] = τ (1 - (k - 1) / p),    k = 1, …, p.

This is a quantum annealing from the mixer to the objective, cut into `p`
steps of length `τ`, and the same schedule as in MFAOA.jl. Because `γ` is
measured in units of the problem's [`fieldscale`](@ref), the same `τ` suits
problems of any size and units. With more layers, the angles change more
slowly from one layer to the next, which usually gives better solutions.
"""
function linear_schedule(p::Integer, τ::Real = 0.5; T::Type{<:AbstractFloat} = Float64)
    p ≥ 1 || throw(ArgumentError("need at least one layer, got p = $p"))
    β = [T(τ) * (1 - T(k - 1) / p) for k in 1:p]
    γ = [T(τ) * T(k) / p for k in 1:p]
    Schedule{T}(β, γ)
end

"""
    interpolate_schedule(sched) -> Schedule

A schedule with one more layer, interpolated linearly from `sched`. Good
angles for `p` layers, interpolated this way, are good starting angles for
`p + 1` layers (the INTERP strategy of Zhou et al., Phys. Rev. X 10, 021067
(2020)): with `θ` standing for `β` or `γ`,

    θ′[k] = (k - 1)/p θ[k - 1] + (p - k + 1)/p θ[k],    k = 1, …, p + 1,

where `θ[0] = θ[p + 1] = 0`.
"""
interpolate_schedule(sched::Schedule) =
    Schedule(_interpolate(sched.β), _interpolate(sched.γ))

"""
    _interpolate(θ) -> Vector

The INTERP rule of [`interpolate_schedule`](@ref) for one angle vector `θ` of
length `p`; returns a vector of length `p + 1`.
"""
function _interpolate(θ::AbstractVector{T}) where {T}
    p = length(θ)
    [T(k - 1) / p * _angle(θ, k - 1) + T(p - k + 1) / p * _angle(θ, k) for k in 1:p+1]
end

"""
    _angle(θ, k)

`θ[k]` for `k` in `1:length(θ)`, and zero for the padding entries `k = 0`
and `k = length(θ) + 1` of the INTERP rule.
"""
_angle(θ::AbstractVector{T}, k::Integer) where {T} = 1 ≤ k ≤ length(θ) ? θ[k] : zero(T)
