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

An optimization problem over spins ``\\sigma_i = \\pm 1``.

### Input
- `h::AbstractVector`: The local fields ``h_i``.
- `J::AbstractMatrix`: The couplings ``J_{ij}``, symmetric with a zero diagonal. `J` can be dense or sparse.
- `Δ` (optional): One positive factor per spin on the mixer angles ``\\beta``. The default, `nothing`, means 1 for every spin.
- `offset::Real=0` (optional): A constant added to the cost function.
- `sense::Symbol=:min` (optional): `:min` to minimize the cost function, `:max` to maximize it.
- `scale` (optional): The unit of the angles ``\\gamma``, see [`fieldscale`](@ref). The default makes the algorithm independent of the units of `h` and `J`; `scale = 1` uses those units themselves.

### Output
- An `IsingProblem` holding all relevant quantities.

### Notes
The cost function is
```math
C(\\sigma) = \\sum_{i=1}^N h_i \\sigma_i + \\sum_{i<j\\leq N} J_{ij} \\sigma_i \\sigma_j + \\mathrm{offset}.
```
Since ``\\sigma_i^2 = 1``, a diagonal entry of `J` would only add a constant; put it into `offset` instead. When the QAOA is measured, qubit ``i`` in the state ``|1\\rangle`` means ``\\sigma_i = +1``, and ``|0\\rangle`` means ``\\sigma_i = -1``.

The arguments mean the same as in [`MFAOA.jl`](https://github.com/Quicopt/MFAOA.jl), which runs the mean-field approximation of the same algorithm: a problem and a [`Schedule`](@ref) written for one package work in the other.
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

An optimization problem over bits ``x_i \\in \\{0, 1\\}``.

### Input
- `Q::AbstractMatrix`: The quadratic weights ``Q_{ij}``, used exactly as given: a symmetric matrix counts each pair of bits twice, an upper-triangular matrix once.
- `c::AbstractVector=zeros(n)` (optional): The linear weights ``c_i``.
- `constant::Real=0` (optional): A constant added to the cost function.
- `sense::Symbol=:min` (optional): `:min` to minimize the cost function, `:max` to maximize it.

### Output
- A `QUBOProblem` holding all relevant quantities.

### Notes
The cost function is
```math
C(x) = \\sum_{i,j=1}^N Q_{ij} x_i x_j + \\sum_{i=1}^N c_i x_i + \\mathrm{constant}.
```
Diagonal entries of `Q` act as linear weights, since ``x_i^2 = x_i``. When the QAOA is measured, qubit ``i`` in the state ``|1\\rangle`` means ``x_i = 1``. The angles ``\\gamma`` are measured in units of the [`fieldscale`](@ref) of the same problem in Ising form, `IsingProblem(prob)`.
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

Returns the QUBO problem `prob` in Ising form, with ``x_i = (1 + \\sigma_i)/2``.

### Input
- `prob::QUBOProblem`: The problem to be converted.
- `Δ`, `scale` (optional): As for [`IsingProblem`](@ref).

### Output
- An `IsingProblem` with the same cost function and sense.

### Notes
The pair terms ``Q_{ij} x_i x_j + Q_{ji} x_j x_i`` become the coupling ``J_{ij} = (Q_{ij} + Q_{ji})/4``. The local fields are ``h_i = (c_i + Q_{ii})/2 + \\sum_j J_{ij}``, and the remaining constants go into the offset.
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

An optimization problem given by the cost function of every bitstring, for cost functions that are not quadratic, such as those of satisfiability problems.

### Input
- `values::AbstractVector`: The cost function of every bitstring of ``n`` bits, so `values` has length ``2^n``. `values[b + 1]` belongs to the bitstring whose bit ``x_i`` is bit `i - 1` of the integer `b`.
- `f`, `n::Integer`: Alternatively, a function `f(x)` returning the cost function of the bitstring `x`, a `Vector{Int8}` of `n ≤ 30` zeros and ones. Note that `f` must not keep `x`, which is reused.
- `sense::Symbol=:min` (optional): `:min` to minimize the cost function, `:max` to maximize it.
- `scale` (optional): The unit of the angles ``\\gamma``, see [`fieldscale`](@ref).

### Output
- A `GeneralProblem` holding all relevant quantities.

### Notes
By default, `scale` is the root mean square, over all bitstrings and bits, of half the change of the cost function when one bit flips. For a quadratic cost function, this equals the [`fieldscale`](@ref) of the problem in Ising form.
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

Returns the number of variables of `prob`, which is the number of qubits of the QAOA.
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

Returns the unit ``s`` in which a [`Schedule`](@ref) measures the angles ``\\gamma`` for `prob`.

### Notes
In layer ``k``, the QAOA multiplies the amplitude of every bitstring by the phase ``e^{-i\\gamma_k C / s}``, where ``C`` is the cost function of that bitstring (with the opposite sign for a minimization). Unless `scale` was given when the problem was built, ``s`` is the root mean square of the local field ``m_i = h_i + \\sum_j J_{ij} \\sigma_j`` over all spins ``i`` and random spins ``\\sigma``,
```math
s = \\sqrt{\\frac{1}{N} \\Big(\\sum_{i=1}^N h_i^2 + \\sum_{i,j=1}^N J_{ij}^2\\Big)}.
```
Multiplying the cost function by a constant multiplies ``s`` by the same constant, so the same angles suit problems in any units.
"""
fieldscale(prob::IsingProblem) = prob.scale
fieldscale(prob::QUBOProblem) = fieldscale(IsingProblem(prob))
fieldscale(prob::GeneralProblem) = prob.scale

"""
    objective(prob::IsingProblem, σ)
    objective(prob::QUBOProblem, x)
    objective(prob::GeneralProblem, x)

Returns the cost function of `prob` for a single bitstring, including the offset or constant.

### Input
- `prob`: The problem.
- `σ::AbstractVector`: The spins, -1 or +1, for an `IsingProblem`.
- `x::AbstractVector`: The bits, 0 or 1, for the other problems.

### Output
- The value of the cost function. Note that maximization problems report the value as it is, without a change of sign.
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
