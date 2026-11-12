# diagonal.jl — the objective of every assignment, stored as the phase per unit angle.
#
# The phase separator of layer k multiplies the amplitude of assignment b by
# exp(-iγ[k] f[b]), where f = ±(C - offset)/s: C is the objective, s the field scale,
# and the sign is + for a maximization. Amplitude index b + 1 belongs to the
# assignment whose qubit i holds bit i - 1 of b.

"""
    DenseDiagonal(f)

The phase per unit angle `f[b+1]` of every stored assignment `b`, as a
`Float64` vector.
"""
struct DenseDiagonal
    f::Vector{Float64}
end

"""
    TableDiagonal(index, values, phases)

The phase per unit angle for an objective with few distinct values:
`f[b+1] == values[index[b+1] + 1]`. It needs two bytes per assignment instead
of eight, and the phase separator computes one complex exponential per
distinct value instead of one per assignment. `phases` holds
`exp(-iγ values)` for the layer at hand; see [`_prepare!`](@ref).
"""
struct TableDiagonal
    index::Vector{UInt16}
    values::Vector{Float64}
    phases::Vector{ComplexF64}
end

"""
    TableDiagonal(index, values)

A table diagonal with room for the phases of every value.
"""
TableDiagonal(index::Vector{UInt16}, values::Vector{Float64}) =
    TableDiagonal(index, values, similar(values, ComplexF64))

"""
    length(d)

The number of stored assignments.
"""
Base.length(d::DenseDiagonal) = length(d.f)
Base.length(d::TableDiagonal) = length(d.index)

"""
    _phase_value(d, b) -> Float64

The phase per unit angle `f[b]` of the stored assignment with (one-based)
index `b`.
"""
@inline _phase_value(d::DenseDiagonal, b::Int) = @inbounds d.f[b]
@inline _phase_value(d::TableDiagonal, b::Int) = @inbounds d.values[d.index[b]+1]

"""
    _prepare!(d, γ)

Fill the phases `exp(-iγ values)` of a [`TableDiagonal`](@ref) for a layer
with angle `γ`. A [`DenseDiagonal`](@ref) computes its phases on the fly, so
there is nothing to do.
"""
_prepare!(::DenseDiagonal, γ::Float64) = nothing
function _prepare!(d::TableDiagonal, γ::Float64)
    @inbounds for t in eachindex(d.values)
        d.phases[t] = cis(-γ * d.values[t])
    end
    nothing
end

"""
    _sense_sign(prob) -> ±1.0

`+1` for a maximization and `-1` for a minimization: the factor that turns the
objective into the quantity the QAOA drives up.
"""
_sense_sign(prob::AbstractProblem) = prob.sense === :max ? 1.0 : -1.0

# --- Ising problems --------------------------------------------------------------

"""
    _diagonal(prob, folded) -> DenseDiagonal or TableDiagonal

The phase per unit angle of every stored assignment of `prob`. With
`folded = true`, only the assignments whose last variable is 0 are stored;
this is exact for a problem whose objective does not change when all variables
flip.
"""
function _diagonal(prob::IsingProblem, folded::Bool)
    h, J = _stored_ising_terms(prob, folded)
    factor = _sense_sign(prob) / fieldscale(prob)
    terms = vcat(h, [J[i, j] for j in axes(J, 2) for i in 1:j-1])
    unit = _grid_unit(terms)
    unit === nothing && return _dense_ising(h, J, factor)
    _table_ising(h, J, unit, factor)
end

"""
    _stored_ising_terms(prob, folded) -> (h, J)

Fields and couplings of the spins that are stored, as dense `Float64` arrays.
Folding fixes the last spin to `σₙ = -1` (bit 0), which turns its couplings
into fields on the other spins: `h′ᵢ = hᵢ - Jᵢₙ`.
"""
function _stored_ising_terms(prob::IsingProblem, folded::Bool)
    h = Vector{Float64}(prob.h)
    J = Matrix{Float64}(prob.J)
    folded || return h, J
    m = length(h) - 1
    h[1:m] .- J[1:m, end], J[1:m, 1:m]
end

"""
    _dense_ising(h, J, factor) -> DenseDiagonal

The Ising objective without offset, times `factor`, for every assignment.
"""
function _dense_ising(h::Vector{Float64}, J::Matrix{Float64}, factor::Float64)
    m = length(h)
    f = Vector{Float64}(undef, 1 << m)
    f[1] = 0.0
    _fill_ising!(f, Vector{Float64}(undef, max(1, 1 << (m - 1))), factor .* h, factor .* J)
    DenseDiagonal(f)
end

"""
    _table_ising(h, J, unit, factor) -> TableDiagonal

The Ising objective as a table. All fields and couplings are integer
multiples of `unit`, so the objective without offset is too, between `-R`
and `R` units with `R = Σ|terms| / unit`. The index of an assignment is its
objective in units plus `R`.
"""
function _table_ising(h::Vector{Float64}, J::Matrix{Float64}, unit::Float64, factor::Float64)
    m = length(h)
    hu = round.(Int32, h ./ unit)
    Ju = round.(Int32, J ./ unit)
    R = sum(abs, hu; init = 0) + sum(abs(Ju[i, j]) for j in 1:m for i in 1:j-1; init = 0)
    index = Vector{UInt16}(undef, 1 << m)
    index[1] = UInt16(R)
    _fill_ising!(index, Vector{Int32}(undef, max(1, 1 << (m - 1))), hu, Ju)
    TableDiagonal(index, [factor * unit * (t - R) for t in 0:2R])
end

"""
    _fill_ising!(c, L, h, J) -> c

Add the Ising objective `Σᵢ h[i] σᵢ + Σ_{i<j} J[i,j] σᵢ σⱼ` to the value
`c[1]` holds on entry, for every assignment of `m = length(h)` spins.
`c[b+1]` belongs to the spins `σᵢ = 2bᵢ - 1`, with `bᵢ` bit `i - 1` of `b`;
`c` has length `2^m`. `L` is scratch space of length `2^(m-1)`.

Adding spin `k` doubles the table, `c(b, σₖ) = c(b) + σₖ Lₖ(b)`, where
`Lₖ(b) = h[k] + Σ_{j<k} J[j,k] σⱼ` is the local field from the earlier spins.
Both steps cost O(2^k), so the whole table costs O(2^m).
"""
function _fill_ising!(c::Vector{T}, L::Vector{S}, h::Vector{S}, J::AbstractMatrix{S}) where {T,S}
    for k in eachindex(h)
        _local_field!(L, h[k], view(J, 1:k-1, k))
        _double_spin!(c, L, 1 << (k - 1))
    end
    c
end

"""
    _local_field!(L, hₖ, Jₖ)

Fill `L[b+1]` with the local field `hₖ + Σⱼ Jₖ[j] σⱼ` on a new spin from the
earlier spins `σⱼ = 2bⱼ - 1`, for all `2^length(Jₖ)` assignments `b` of
those spins. Turning spin `j` from -1 to +1 adds `2Jₖ[j]`.
"""
function _local_field!(L::Vector{S}, hk::S, Jk::AbstractVector{S}) where {S}
    L[1] = hk - sum(Jk; init = zero(S))
    for j in eachindex(Jk)
        s = 1 << (j - 1)
        twoJ = Jk[j] + Jk[j]
        @inbounds @simd for b in 1:s
            L[b+s] = L[b] + twoJ
        end
    end
    L
end

"""
    _double_spin!(c, L, half)

Extend the table `c[1:half]` of the earlier spins by a new spin with local
field `L`: `c[b + half] = c[b] + L[b]` for the new spin at +1 (bit 1), and
`c[b] = c[b] - L[b]` for the new spin at -1 (bit 0).
"""
function _double_spin!(c::Vector{T}, L::Vector{S}, half::Int) where {T,S}
    @inbounds @simd for b in 1:half
        cb = c[b]
        c[b+half] = _plus(cb, L[b])
        c[b] = _plus(cb, -L[b])
    end
    c
end

"""
    _plus(c, l)

`c + l` in the storage type of `c`: a plain sum for `Float64`, and for a
`UInt16` table index (which never leaves `0:65535` here) a sum in `Int32`
converted back.
"""
@inline _plus(c::Float64, l::Float64) = c + l
@inline _plus(c::UInt16, l::Int32) = (Int32(c) + l) % UInt16

"""
    _grid_unit(terms) -> unit or nothing

The largest unit among `1, 1/2, …, 1/16` that makes every term an integer, if
the integer terms add up to at most 32767 in absolute value; `nothing`
otherwise. Then the objective takes at most 65535 values, which a
[`TableDiagonal`](@ref) stores.
"""
function _grid_unit(terms::AbstractVector{<:Real})
    for unit in (1.0, 0.5, 0.25, 0.125, 0.0625)
        all(t -> isinteger(t / unit), terms) || continue
        sum(t -> abs(t / unit), terms; init = 0.0) ≤ 32767 && return unit
        return nothing
    end
    nothing
end

# --- QUBO problems ---------------------------------------------------------------

"""
    _diagonal(prob::QUBOProblem, folded) -> DenseDiagonal or TableDiagonal

The phase per unit angle of every stored assignment `x` of a QUBO problem.
Folding fixes the last variable to 0, which removes it.
"""
function _diagonal(prob::QUBOProblem, folded::Bool)
    linear, pair = _stored_qubo_terms(prob, folded)
    factor = _sense_sign(prob) / fieldscale(prob)
    terms = vcat(linear, [pair[i, j] for j in axes(pair, 2) for i in 1:j-1])
    unit = _grid_unit(terms)
    unit === nothing && return _dense_qubo(linear, pair, factor)
    _table_qubo(linear, pair, unit, factor)
end

"""
    _stored_qubo_terms(prob, folded) -> (linear, pair)

The weight `c[i] + Q[i,i]` of each variable alone and the weight
`Q[i,j] + Q[j,i]` of each pair, for the stored variables, as dense `Float64`
arrays.
"""
function _stored_qubo_terms(prob::QUBOProblem, folded::Bool)
    Q = Matrix{Float64}(prob.Q)
    m = length(prob.c) - folded
    linear = [prob.c[i] + Q[i, i] for i in 1:m]
    pair = [Q[i, j] + Q[j, i] for i in 1:m, j in 1:m]
    linear, pair
end

"""
    _dense_qubo(linear, pair, factor) -> DenseDiagonal

The QUBO objective without constant, times `factor`, for every assignment.
"""
function _dense_qubo(linear::Vector{Float64}, pair::Matrix{Float64}, factor::Float64)
    m = length(linear)
    f = Vector{Float64}(undef, 1 << m)
    f[1] = 0.0
    _fill_qubo!(f, Vector{Float64}(undef, max(1, 1 << (m - 1))), factor .* linear, factor .* pair)
    DenseDiagonal(f)
end

"""
    _table_qubo(linear, pair, unit, factor) -> TableDiagonal

The QUBO objective as a table; see [`_table_ising`](@ref).
"""
function _table_qubo(linear::Vector{Float64}, pair::Matrix{Float64}, unit::Float64, factor::Float64)
    m = length(linear)
    lu = round.(Int32, linear ./ unit)
    pu = round.(Int32, pair ./ unit)
    R = sum(abs, lu; init = 0) + sum(abs(pu[i, j]) for j in 1:m for i in 1:j-1; init = 0)
    index = Vector{UInt16}(undef, 1 << m)
    index[1] = UInt16(R)
    _fill_qubo!(index, Vector{Int32}(undef, max(1, 1 << (m - 1))), lu, pu)
    TableDiagonal(index, [factor * unit * (t - R) for t in 0:2R])
end

"""
    _fill_qubo!(c, L, linear, pair) -> c

Add the QUBO objective `Σᵢ linear[i] xᵢ + Σ_{i<j} pair[i,j] xᵢ xⱼ` to the value
`c[1]` holds on entry, for every assignment `x` of `m = length(linear)` bits;
`c[b+1]` belongs to `xᵢ` = bit `i - 1` of `b`. Like [`_fill_ising!`](@ref),
adding variable `k` doubles the table: `c(b, 1) = c(b) + Lₖ(b)` with
`Lₖ(b) = linear[k] + Σ_{j<k} pair[j,k] xⱼ`, and `c(b, 0) = c(b)`.
"""
function _fill_qubo!(c::Vector{T}, L::Vector{S}, linear::Vector{S}, pair::AbstractMatrix{S}) where {T,S}
    for k in eachindex(linear)
        _local_weight!(L, linear[k], view(pair, 1:k-1, k))
        half = 1 << (k - 1)
        @inbounds @simd for b in 1:half
            c[b+half] = _plus(c[b], L[b])
        end
    end
    c
end

"""
    _local_weight!(L, wₖ, pairₖ)

Fill `L[b+1]` with the weight `wₖ + Σⱼ pairₖ[j] xⱼ` of a new variable for all
assignments `b` of the earlier variables. Setting `xⱼ = 1` adds `pairₖ[j]`.
"""
function _local_weight!(L::Vector{S}, wk::S, pairk::AbstractVector{S}) where {S}
    L[1] = wk
    for j in eachindex(pairk)
        s = 1 << (j - 1)
        @inbounds @simd for b in 1:s
            L[b+s] = L[b] + pairk[j]
        end
    end
    L
end

# --- general problems ------------------------------------------------------------

"""
    _diagonal(prob::GeneralProblem, folded) -> DenseDiagonal or TableDiagonal

The tabulated objective times `±1/s`, as a table when
[`_value_unit`](@ref) finds a unit for it.
"""
function _diagonal(prob::GeneralProblem, folded::Bool)
    v = folded ? prob.values[1:end÷2] : prob.values
    factor = _sense_sign(prob) / fieldscale(prob)
    unit = _value_unit(v)
    unit === nothing && return DenseDiagonal(Float64.(factor .* v))
    low = minimum(v)
    span = round(Int, (maximum(v) - low) / unit)
    index = [UInt16((x - low) / unit) for x in v]
    TableDiagonal(index, [factor * (low + unit * t) for t in 0:span])
end

"""
    _value_unit(v) -> unit or nothing

The largest unit among `1, 1/2, …, 1/16` that makes every value an integer,
if the values then span at most 65534 units; `nothing` otherwise.
"""
function _value_unit(v::AbstractVector{<:Real})
    low, high = extrema(v)
    for unit in (1.0, 0.5, 0.25, 0.125, 0.0625)
        all(x -> isinteger(x / unit), v) || continue
        return (high - low) / unit ≤ 65534 ? unit : nothing
    end
    nothing
end
