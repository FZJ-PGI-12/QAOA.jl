# problems.jl — the optimization problems QAOA.jl accepts.

"""
    AbstractProblem

Supertype of the problem types [`IsingProblem`](@ref), [`QUBOProblem`](@ref)
and [`GeneralProblem`](@ref).
"""
abstract type AbstractProblem end

"""The two optimization senses a problem can have."""
const _SENSES = (:min, :max)

"""
    _check_sense(sense)

Throw an `ArgumentError` unless `sense` is `:min` or `:max`.
"""
_check_sense(sense::Symbol) =
    sense in _SENSES || throw(ArgumentError("sense must be :min or :max, got :$sense"))

"""
    _entries(A)

The entries of `A` that can be nonzero: all of them for a dense matrix, the
stored values for a sparse one.
"""
_entries(A::AbstractMatrix) = A
_entries(A::SparseMatrixCSC) = nonzeros(A)

"""
    _check_couplings(J, n)

Throw an `ArgumentError` unless `J` is `n × n`, zero on the diagonal and
symmetric.
"""
function _check_couplings(J::AbstractMatrix, n::Integer)
    size(J) == (n, n) ||
        throw(ArgumentError("J must be $(n)×$(n) to match h, got $(size(J))"))
    for i in 1:n
        iszero(J[i, i]) || throw(ArgumentError(
            "J must be zero on the diagonal, but J[$i,$i] = $(J[i, i]). Since σ[i]^2 = 1, " *
            "a diagonal entry only adds a constant: add it to offset instead."))
    end
    issymmetric(J) || throw(ArgumentError("J must be symmetric"))
    return nothing
end

"""
    _check_factors(Δ, n)

Throw an `ArgumentError` unless `Δ` is `nothing` or holds `n` positive mixer
factors.
"""
function _check_factors(Δ, n::Integer)
    Δ === nothing && return nothing
    length(Δ) == n ||
        throw(ArgumentError("Δ must have one entry per variable, got $(length(Δ)) for n = $n"))
    all(>(0), Δ) || throw(ArgumentError("all mixer factors Δ[i] must be positive"))
    return nothing
end

"""
    IsingProblem(h, J; Δ = nothing, offset = 0, sense = :min, scale = nothing)

An optimization problem over spins `σ[i]` that are -1 or +1: minimize (or, with
`sense = :max`, maximize)

    h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j] + offset.

`J` must be symmetric with a zero diagonal. Since `σ[i]^2 = 1`, a diagonal
entry would only add a constant; put it into `offset` instead. `J` can be
dense or sparse.

When the QAOA is measured, qubit `i` in state |1⟩ means `σ[i] = +1`, and |0⟩
means `σ[i] = -1`.

Two keywords tune the algorithm and rarely need to be set:

- `Δ`: one positive factor per spin on the mixer angles `β` of a
  [`Schedule`](@ref). The default, `nothing`, means 1 for every spin.
- `scale`: the unit of the angles `γ`; see [`fieldscale`](@ref). The default
  makes the algorithm independent of the units of `h` and `J`; `scale = 1`
  uses those units themselves.

The arguments mean the same as in [MFAOA.jl](https://github.com/Quicopt/MFAOA.jl),
which runs the mean-field approximation of the same algorithm: a problem and a
[`Schedule`](@ref) written for one package work in the other.
"""
struct IsingProblem{T<:AbstractFloat,M<:AbstractMatrix{T},D<:Union{Nothing,Vector{T}}} <: AbstractProblem
    h::Vector{T}
    J::M
    Δ::D
    offset::T
    sense::Symbol
    scale::T
end

function IsingProblem(h::AbstractVector, J::AbstractMatrix;
                      Δ::Union{Nothing,AbstractVector} = nothing,
                      offset::Real = 0, sense::Symbol = :min,
                      scale::Union{Nothing,Real} = nothing)
    _check_sense(sense)
    n = length(h)
    n ≥ 1 || throw(ArgumentError("the problem needs at least one variable"))
    _check_couplings(J, n)
    _check_factors(Δ, n)
    T = float(promote_type(eltype(h), eltype(J), typeof(offset),
                           Δ === nothing ? Bool : eltype(Δ)))
    hT = convert(Vector{T}, h)
    JT = convert(AbstractMatrix{T}, J)
    ΔT = Δ === nothing ? nothing : convert(Vector{T}, Δ)
    s = scale === nothing ? _rms_field(hT, JT) : T(scale)
    scale === nothing && iszero(s) && (s = one(T))      # an objective that is constant
    s > 0 || throw(ArgumentError("scale must be positive, got $scale"))
    IsingProblem{T,typeof(JT),typeof(ΔT)}(hT, JT, ΔT, T(offset), sense, s)
end

"""
    _rms_field(h, J) -> s

The root mean square of the local field `m[i] = h[i] + Σⱼ J[i,j] σ[j]` over
all spins `i` and uniformly random spins `σ`. The cross terms average out,
which leaves `s = sqrt((Σᵢ hᵢ² + Σᵢⱼ Jᵢⱼ²) / n)`.
"""
_rms_field(h::AbstractVector, J::AbstractMatrix) =
    sqrt((sum(abs2, h) + sum(abs2, _entries(J))) / length(h))

"""
    QUBOProblem(Q; c = zeros(n), constant = 0, sense = :min)

An optimization problem over binary variables `x[i]` that are 0 or 1: minimize
(or, with `sense = :max`, maximize)

    x' * Q * x + c' * x + constant.

`Q` is used exactly as given: a symmetric matrix counts each pair of
variables twice, an upper-triangular matrix once. Diagonal entries act as
linear terms, since `x[i]^2 = x[i]`. When the QAOA is measured, qubit `i` in
state |1⟩ means `x[i] = 1`.

The angles `γ` are measured in units of the [`fieldscale`](@ref) of the same
problem in Ising form, [`IsingProblem(prob)`](@ref IsingProblem(::QUBOProblem)).
"""
struct QUBOProblem{T<:AbstractFloat,M<:AbstractMatrix{T}} <: AbstractProblem
    Q::M
    c::Vector{T}
    constant::T
    sense::Symbol
end

function QUBOProblem(Q::AbstractMatrix; c::AbstractVector = zeros(Bool, size(Q, 1)),
                     constant::Real = 0, sense::Symbol = :min)
    _check_sense(sense)
    n = size(Q, 1)
    n ≥ 1 || throw(ArgumentError("the problem needs at least one variable"))
    size(Q) == (n, n) || throw(ArgumentError("Q must be square, got $(size(Q))"))
    length(c) == n ||
        throw(ArgumentError("c must have one entry per variable, got $(length(c)) for n = $n"))
    T = float(promote_type(eltype(Q), eltype(c), typeof(constant)))
    QT = convert(AbstractMatrix{T}, Q)
    QUBOProblem{T,typeof(QT)}(QT, convert(Vector{T}, c), T(constant), sense)
end

"""
    IsingProblem(prob::QUBOProblem; Δ = nothing, scale = nothing)

The same problem in Ising form, with `x = (1 + σ) / 2`. The pair terms
`Q[i,j] x[i] x[j] + Q[j,i] x[j] x[i]` become the coupling
`J[i,j] = (Q[i,j] + Q[j,i]) / 4`. The weight of `x[i]` alone, `c[i] + Q[i,i]`,
goes half into `h[i]` and half into the offset.
"""
function IsingProblem(prob::QUBOProblem{T}; Δ::Union{Nothing,AbstractVector} = nothing,
                      scale::Union{Nothing,Real} = nothing) where {T}
    n = nvariables(prob)
    J = _ising_couplings(prob.Q)
    linear = [prob.c[i] + prob.Q[i, i] for i in 1:n]
    coupling_sums = J * ones(T, n)
    h = linear ./ 2 .+ coupling_sums
    offset = prob.constant + sum(linear) / 2 + sum(coupling_sums) / 2
    IsingProblem(h, J; Δ, offset, sense = prob.sense, scale)
end

"""
    _ising_couplings(Q) -> J

The Ising couplings `J[i,j] = (Q[i,j] + Q[j,i]) / 4` for `i ≠ j`, with a zero
diagonal; sparse if `Q` is.
"""
function _ising_couplings(Q::AbstractMatrix{T}) where {T}
    n = size(Q, 1)
    [i == j ? zero(T) : (Q[i, j] + Q[j, i]) / 4 for i in 1:n, j in 1:n]
end

function _ising_couplings(Q::SparseMatrixCSC{T}) where {T}
    rows, cols, vals = findnz(Q)
    keep = rows .!= cols
    i, j, v = rows[keep], cols[keep], vals[keep] ./ 4
    sparse(vcat(i, j), vcat(j, i), vcat(v, v), size(Q)...)
end

"""
    GeneralProblem(values; sense = :min, scale = nothing)
    GeneralProblem(f, n; sense = :min, scale = nothing)

An optimization problem given by the objective value of every assignment of
`n` binary variables, for objectives that are not quadratic, such as
satisfiability problems.

`values[b + 1]` is the objective of the assignment whose variable `x[i]` is
bit `i - 1` of the integer `b`, so `values` has length `2^n`. The second form
evaluates `f(x)` for every assignment, with `x` a `Vector{Int8}` of zeros and
ones. `f` must not keep `x`, which is reused.

`scale` is the unit of the angles `γ`. By default, it is the root mean square,
over all assignments and variables, of half the change of the objective when
one variable flips. For a quadratic objective, this equals the
[`fieldscale`](@ref) of the problem in Ising form.
"""
struct GeneralProblem{T<:AbstractFloat} <: AbstractProblem
    values::Vector{T}
    n::Int
    sense::Symbol
    scale::T
end

function GeneralProblem(values::AbstractVector{<:Real}; sense::Symbol = :min,
                        scale::Union{Nothing,Real} = nothing)
    _check_sense(sense)
    N = length(values)
    (N ≥ 2 && ispow2(N)) ||
        throw(ArgumentError("values must have length 2^n with n ≥ 1, got $N"))
    T = float(eltype(values))
    v = convert(Vector{T}, values)
    n = trailing_zeros(N)
    s = scale === nothing ? _rms_flip(v, n) : T(scale)
    scale === nothing && iszero(s) && (s = one(T))
    s > 0 || throw(ArgumentError("scale must be positive, got $scale"))
    GeneralProblem{T}(v, n, sense, s)
end

function GeneralProblem(f, n::Integer; sense::Symbol = :min, scale::Union{Nothing,Real} = nothing)
    1 ≤ n ≤ 30 || throw(ArgumentError("n must be between 1 and 30, got $n"))
    x = zeros(Int8, n)
    values = map(0:(1 << n)-1) do b
        for i in 1:n
            x[i] = (b >> (i - 1)) & 1
        end
        f(x)
    end
    GeneralProblem(values; sense, scale)
end

"""
    _rms_flip(v, n) -> s

The root mean square, over all `2^n` assignments `b` and variables `i`, of
half the change `(v[b] - v[b with bit i flipped]) / 2` of the tabulated
objective `v`. For a quadratic objective, this is [`_rms_field`](@ref) of its
Ising form: flipping `σᵢ` changes the objective by `2σᵢ mᵢ`.
"""
function _rms_flip(v::Vector{T}, n::Int) where {T}
    acc = zero(T)
    @inbounds for k in 0:n-1
        s = 1 << k
        for base in 0:2s:length(v)-1, b in base+1:base+s
            acc += abs2((v[b] - v[b+s]) / 2)
        end
    end
    sqrt(2acc / (n * length(v)))
end

"""
    nvariables(prob)

The number of binary variables of `prob`, which is the number of qubits.
"""
nvariables(prob::IsingProblem) = length(prob.h)
nvariables(prob::QUBOProblem) = length(prob.c)
nvariables(prob::GeneralProblem) = prob.n

"""
    eltype(prob)

The floating-point type the problem is stored in.
"""
Base.eltype(::IsingProblem{T}) where {T} = T
Base.eltype(::QUBOProblem{T}) where {T} = T
Base.eltype(::GeneralProblem{T}) where {T} = T

"""
    fieldscale(prob) -> s

The unit in which a [`Schedule`](@ref) measures the angles `γ` for `prob`.

In layer `k`, the QAOA multiplies the amplitude of every assignment by a phase
`exp(-iγ[k] C / s)`, where `C` is the objective of that assignment (with the
opposite sign for a minimization). Unless `scale` was given when the problem
was built, `s` is the root mean square of the local field
`m[i] = h[i] + Σⱼ J[i,j] σ[j]` over all spins `i` and random spins `σ`:

    s = sqrt((Σᵢ hᵢ² + Σᵢⱼ Jᵢⱼ²) / n).

Multiplying the objective by a constant multiplies `s` by the same constant,
so the same angles suit problems in any units.
"""
fieldscale(prob::IsingProblem) = prob.scale
fieldscale(prob::QUBOProblem) = fieldscale(IsingProblem(prob))
fieldscale(prob::GeneralProblem) = prob.scale

"""
    objective(prob::IsingProblem, σ)
    objective(prob::QUBOProblem, x)
    objective(prob::GeneralProblem, x)

The objective value of an assignment, including the offset or constant. `σ`
holds spins -1 and +1, `x` holds bits 0 and 1. Maximization problems report
the value as stated, without a sign change.
"""
objective(prob::IsingProblem, σ::AbstractVector) =
    dot(prob.h, σ) + dot(σ, prob.J, σ) / 2 + prob.offset
objective(prob::QUBOProblem, x::AbstractVector) =
    dot(x, prob.Q, x) + dot(prob.c, x) + prob.constant
function objective(prob::GeneralProblem, x::AbstractVector)
    b = 0
    for i in eachindex(x)
        b |= Int(x[i]) << (i - firstindex(x))
    end
    prob.values[b+1]
end

"""
    _flipsymmetric(prob) -> Bool

Whether flipping all variables at once leaves the objective unchanged, exactly:
an Ising form without fields, or a table with equal values for every
assignment and its complement.
"""
_flipsymmetric(prob::IsingProblem) = all(iszero, prob.h)
_flipsymmetric(prob::QUBOProblem) = _flipsymmetric(IsingProblem(prob))
function _flipsymmetric(prob::GeneralProblem)
    v = prob.values
    N = length(v)
    all(b -> v[b] == v[N+1-b], 1:N÷2)
end

"""
    _offset(prob)

The constant part of the objective, which the phase separator leaves out: the
offset of an Ising problem, the constant of a QUBO problem, zero otherwise.
"""
_offset(prob::IsingProblem) = prob.offset
_offset(prob::QUBOProblem) = prob.constant
_offset(prob::GeneralProblem) = zero(eltype(prob))

"""
    _factors(prob)

The mixer factors `Δ` of an Ising problem, or `nothing` (all 1).
"""
_factors(prob::IsingProblem) = prob.Δ
_factors(::AbstractProblem) = nothing

"""
    show(io, prob)

Print the problem type, the number of variables and the sense.
"""
function Base.show(io::IO, prob::IsingProblem{T}) where {T}
    print(io, "IsingProblem{$T}: $(nvariables(prob)) spins, sense = :$(prob.sense)")
end
function Base.show(io::IO, prob::QUBOProblem{T}) where {T}
    print(io, "QUBOProblem{$T}: $(nvariables(prob)) variables, sense = :$(prob.sense)")
end
function Base.show(io::IO, prob::GeneralProblem{T}) where {T}
    print(io, "GeneralProblem{$T}: $(nvariables(prob)) variables, sense = :$(prob.sense)")
end
