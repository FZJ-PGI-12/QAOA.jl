# kernels.jl — loops over the amplitudes of the state vector.
#
# Qubit q (one-based) is bit q - 1 of the amplitude index b: the two amplitudes of
# assignments that differ only in qubit q are `stride = 2^(q-1)` apart. A "turn" of
# qubit q by the angle θ is the rotation exp(-iθX) on every such pair.

"""
Number of low qubits that are turned inside blocks of `2^BLOCK_BITS` amplitudes
(32 KiB), which stay in the cache while all low qubits are turned.
"""
const BLOCK_BITS = 11

"""
States with fewer amplitudes than this run on one thread: below it, starting
threads costs more than it saves.
"""
const THREADING_THRESHOLD = 1 << 15

# --- scalar helpers ----------------------------------------------------------------

"""
    _sincos(x) -> (sin(x), cos(x))

Sine and cosine without branches, so that loops calling it vectorize; about
four times faster than `sincos` in such loops. The argument is reduced to
`r ∈ [-π/4, π/4]` with a two-term `π/2`, and Taylor polynomials of degree 15
and 16 give an error below `3e-16` for `|x| ≲ 1e6`.
"""
@inline function _sincos(x::Float64)
    q = round(x * 0.6366197723675814)                      # x / (π/2)
    r = muladd(-q, 1.5707963267948966, x)
    r = muladd(-q, 6.123233995736766e-17, r)
    r2 = r * r
    sin_r = r * evalpoly(r2, (1.0, -1 / 6, 1 / 120, -1 / 5040, 1 / 362880, -1 / 39916800,
                              1 / 6227020800, -1 / 1307674368000))
    cos_r = evalpoly(r2, (1.0, -1 / 2, 1 / 24, -1 / 720, 1 / 40320, -1 / 3628800,
                          1 / 479001600, -1 / 87178291200, 1 / 20922789888000))
    quadrant = unsafe_trunc(Int64, q) & 3
    s = ifelse(quadrant == 0, sin_r, ifelse(quadrant == 1, cos_r, ifelse(quadrant == 2, -sin_r, -cos_r)))
    c = ifelse(quadrant == 0, cos_r, ifelse(quadrant == 1, -sin_r, ifelse(quadrant == 2, -cos_r, sin_r)))
    return s, c
end

"""
    _turn(a, b, c, s) -> (a′, b′)

The amplitudes `(a, b)` of two assignments that differ in one qubit, after
the rotation `exp(-iθX) = cos θ - i sin θ X` of that qubit, with `c = cos θ`
and `s = sin θ`: `a′ = c a - i s b` and `b′ = c b - i s a`.
"""
@inline function _turn(a::ComplexF64, b::ComplexF64, c::Float64, s::Float64)
    return ComplexF64(c * real(a) + s * imag(b), c * imag(a) - s * real(b)),
           ComplexF64(c * real(b) + s * imag(a), c * imag(b) - s * real(a))
end

"""
    _imdot(l, a) -> Float64

`Im(conj(l) * a)`, the imaginary part of one term of the inner product `⟨λ|ψ⟩`.
"""
@inline _imdot(l::ComplexF64, a::ComplexF64) = real(l) * imag(a) - imag(l) * real(a)

"""
    _rotate(a, s, c) -> ComplexF64

`a * (c + i s)`, the amplitude `a` times the phase with sine `s` and cosine `c`.
"""
@inline _rotate(a::ComplexF64, s::Float64, c::Float64) =
    ComplexF64(c * real(a) - s * imag(a), c * imag(a) + s * real(a))

"""
    _add(a, b)

`a + b` for numbers and, entry by entry, for pairs of numbers: the reduction of
[`_chunked_sum`](@ref).
"""
@inline _add(a::Float64, b::Float64) = a + b
@inline _add(a::NTuple{2,Float64}, b::NTuple{2,Float64}) = (a[1] + b[1], a[2] + b[2])

# --- threading -----------------------------------------------------------------------

"""
    _chunks(work!, len, threaded)

Call `work!(lo, hi)` on consecutive ranges that cover `1:len`: a single range,
or one range per thread if `threaded`.
"""
function _chunks(work!::F, len::Int, threaded::Bool) where {F}
    nt = threaded ? min(nthreads(), len) : 1
    if nt == 1
        work!(1, len)
    else
        @threads :static for t in 1:nt
            work!((t - 1) * len ÷ nt + 1, t * len ÷ nt)
        end
    end
    nothing
end

"""
    _chunked_sum(work, len, threaded, zero) -> sum

Like [`_chunks`](@ref), but `work(lo, hi)` returns a number or a pair of
numbers, and the results of all ranges are added up, starting from `zero`.
The ranges are fixed by `len` and the number of threads, so the result is
reproducible.
"""
function _chunked_sum(work::F, len::Int, threaded::Bool, zero::T) where {F,T}
    nt = threaded ? min(nthreads(), len) : 1
    nt == 1 && return _add(zero, work(1, len))
    partial = fill(zero, nt)
    @threads :static for t in 1:nt
        partial[t] = work((t - 1) * len ÷ nt + 1, t * len ÷ nt)
    end
    foldl(_add, partial; init = zero)
end

"""
    _runs(work, stride, groupsize, t0, t1, zero) -> sum

Split the first indices number `t0:t1` of a pass into runs of contiguous
indices and add up `work(lo, hi)` over the runs, starting from `zero`.

A pass over qubit `q` (and `q + 1`) visits the amplitudes in groups of
`groupsize` (`2stride` or `4stride`); the first index of a pair (or
quadruple) is one of the first `stride` amplitudes of its group. Counted
`t = 0, 1, …`, first index `t` lies in group `t ÷ stride` at offset
`t % stride`.
"""
function _runs(work::F, stride::Int, groupsize::Int, t0::Int, t1::Int, zero::Float64) where {F}
    total = zero
    t = t0
    while t ≤ t1
        group, offset = divrem(t, stride)
        last = min(stride - 1, offset + t1 - t)
        lo = group * groupsize + offset + 1
        total += work(lo, lo + last - offset)
        t += last - offset + 1
    end
    total
end

# --- phase separator ---------------------------------------------------------------

"""
    _phase!(ψ, d, γ, lo, hi)

Multiply the amplitudes `ψ[lo:hi]` by `exp(-iγ f)`. A [`TableDiagonal`](@ref)
must hold the phases for `γ`; see [`_prepare!`](@ref).
"""
@inline function _phase!(ψ::Vector{ComplexF64}, d::DenseDiagonal, γ::Float64, lo::Int, hi::Int)
    f = d.f
    @inbounds @simd for b in lo:hi
        s, c = _sincos(-γ * f[b])
        ψ[b] = _rotate(ψ[b], s, c)
    end
end

@inline function _phase!(ψ::Vector{ComplexF64}, d::TableDiagonal, γ::Float64, lo::Int, hi::Int)
    index = d.index
    phases = d.phases
    @inbounds @simd for b in lo:hi
        ψ[b] *= phases[index[b]+1]
    end
end

"""
    _phase_adjoint!(ψ, λ, d, γ, lo, hi) -> Σ f Im(conj(λ) ψ)

Undo the phase separator on `ψ[lo:hi]` and `λ[lo:hi]` (multiply both by
`exp(+iγ f)`) and return this range's part of `Im⟨λ|f|ψ⟩`, from which the
derivative with respect to `γ` follows. A [`TableDiagonal`](@ref) must hold
the phases for `-γ`.
"""
@inline function _phase_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, d::DenseDiagonal,
                                 γ::Float64, lo::Int, hi::Int)
    f = d.f
    g = 0.0
    @inbounds @simd for b in lo:hi
        g += f[b] * _imdot(λ[b], ψ[b])
        s, c = _sincos(γ * f[b])
        ψ[b] = _rotate(ψ[b], s, c)
        λ[b] = _rotate(λ[b], s, c)
    end
    g
end

@inline function _phase_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, d::TableDiagonal,
                                 γ::Float64, lo::Int, hi::Int)
    index = d.index
    values = d.values
    phases = d.phases
    g = 0.0
    @inbounds @simd for b in lo:hi
        t = index[b] + 1
        g += values[t] * _imdot(λ[b], ψ[b])
        ψ[b] *= phases[t]
        λ[b] *= phases[t]
    end
    g
end

# --- turning qubits: runs of contiguous pairs -------------------------------------

"""
    _pair_run!(ψ, stride, lo, hi, c, s)

Turn the pairs `(ψ[b], ψ[b + stride])` for `b` in `lo:hi`.
"""
@inline function _pair_run!(ψ::Vector{ComplexF64}, stride::Int, lo::Int, hi::Int, c::Float64, s::Float64)
    @inbounds @simd for b in lo:hi
        ψ[b], ψ[b+stride] = _turn(ψ[b], ψ[b+stride], c, s)
    end
    0.0
end

"""
    _pair_run_adjoint!(ψ, λ, stride, lo, hi, c, s) -> Σ Im⟨λ|X|ψ⟩

Undo a turn (`s` is the negated sine) on the pairs at `b` and `b + stride`,
`b ∈ lo:hi`, of both `ψ` and `λ`, and return their part of `Im⟨λ|X|ψ⟩` for
this qubit. The value does not depend on whether the turn is undone before or
after: `X` commutes with the turn, and both vectors get the same one.
"""
@inline function _pair_run_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, stride::Int,
                                    lo::Int, hi::Int, c::Float64, s::Float64)
    g = 0.0
    @inbounds @simd for b in lo:hi
        a0 = ψ[b]; a1 = ψ[b+stride]; l0 = λ[b]; l1 = λ[b+stride]
        g += _imdot(l0, a1) + _imdot(l1, a0)
        ψ[b], ψ[b+stride] = _turn(a0, a1, c, s)
        λ[b], λ[b+stride] = _turn(l0, l1, c, s)
    end
    g
end

"""
    _quad_run!(ψ, stride, lo, hi, c1, s1, c2, s2)

Turn two neighbouring qubits in one pass: the qubit with stride `stride`
(cosine `c1`, sine `s1`) and the one with stride `2stride` (`c2`, `s2`), on
the quadruples starting at `b ∈ lo:hi`. One pass instead of two halves the
memory traffic of the qubits that do not fit into a block.
"""
@inline function _quad_run!(ψ::Vector{ComplexF64}, stride::Int, lo::Int, hi::Int,
                            c1::Float64, s1::Float64, c2::Float64, s2::Float64)
    @inbounds @simd for b in lo:hi
        a0, a1, a2, a3 = ψ[b], ψ[b+stride], ψ[b+2stride], ψ[b+3stride]
        a0, a1 = _turn(a0, a1, c1, s1)
        a2, a3 = _turn(a2, a3, c1, s1)
        a0, a2 = _turn(a0, a2, c2, s2)
        a1, a3 = _turn(a1, a3, c2, s2)
        ψ[b], ψ[b+stride], ψ[b+2stride], ψ[b+3stride] = a0, a1, a2, a3
    end
    0.0
end

"""
    _quad_run_adjoint!(ψ, λ, stride, lo, hi, c1, s1, c2, s2, w1, w2) -> g

Undo the turns of [`_quad_run!`](@ref) on `ψ` and `λ` (`s1`, `s2` negated)
and return `w1 Im⟨λ|X₁|ψ⟩ + w2 Im⟨λ|X₂|ψ⟩` for the two qubits on these
quadruples, with `w` the qubits' mixer factors.
"""
@inline function _quad_run_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, stride::Int,
                                    lo::Int, hi::Int, c1::Float64, s1::Float64, c2::Float64,
                                    s2::Float64, w1::Float64, w2::Float64)
    g = 0.0
    @inbounds @simd for b in lo:hi
        a0, a1, a2, a3 = ψ[b], ψ[b+stride], ψ[b+2stride], ψ[b+3stride]
        l0, l1, l2, l3 = λ[b], λ[b+stride], λ[b+2stride], λ[b+3stride]
        g += w1 * (_imdot(l0, a1) + _imdot(l1, a0) + _imdot(l2, a3) + _imdot(l3, a2))
        g += w2 * (_imdot(l0, a2) + _imdot(l2, a0) + _imdot(l1, a3) + _imdot(l3, a1))
        a0, a1 = _turn(a0, a1, c1, s1); a2, a3 = _turn(a2, a3, c1, s1)
        a0, a2 = _turn(a0, a2, c2, s2); a1, a3 = _turn(a1, a3, c2, s2)
        l0, l1 = _turn(l0, l1, c1, s1); l2, l3 = _turn(l2, l3, c1, s1)
        l0, l2 = _turn(l0, l2, c2, s2); l1, l3 = _turn(l1, l3, c2, s2)
        ψ[b], ψ[b+stride], ψ[b+2stride], ψ[b+3stride] = a0, a1, a2, a3
        λ[b], λ[b+stride], λ[b+2stride], λ[b+3stride] = l0, l1, l2, l3
    end
    g
end

"""
    _fold_run!(ψ, lo, hi, c, s)

Turn the folded qubit on the pairs `(ψ[b], ψ[N + 1 - b])`, `b ∈ lo:hi`, where
`N = length(ψ)`.

A folded state stores only the assignments whose last qubit is 0. Flipping
the last qubit of stored assignment `b - 1` gives an assignment that is not
stored, but its amplitude equals that of its complement, which is stored at
index `N + 1 - b`. Turning the last qubit therefore pairs `b` with `N + 1 - b`.
"""
@inline function _fold_run!(ψ::Vector{ComplexF64}, lo::Int, hi::Int, c::Float64, s::Float64)
    N = length(ψ)
    @inbounds for b in lo:hi
        ψ[b], ψ[N+1-b] = _turn(ψ[b], ψ[N+1-b], c, s)
    end
    0.0
end

"""
    _fold_run_adjoint!(ψ, λ, lo, hi, c, s) -> Σ Im⟨λ|X|ψ⟩

Undo the turn of the folded qubit (see [`_fold_run!`](@ref)) on `ψ` and `λ`
and return this range's part of `Im⟨λ|X|ψ⟩` for that qubit.
"""
@inline function _fold_run_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, lo::Int, hi::Int,
                                    c::Float64, s::Float64)
    N = length(ψ)
    g = 0.0
    @inbounds for b in lo:hi
        a0 = ψ[b]; a1 = ψ[N+1-b]; l0 = λ[b]; l1 = λ[N+1-b]
        g += _imdot(l0, a1) + _imdot(l1, a0)
        ψ[b], ψ[N+1-b] = _turn(a0, a1, c, s)
        λ[b], λ[N+1-b] = _turn(l0, l1, c, s)
    end
    g
end

"""
    _insert_zero(t, position) -> Int

`t` with a zero bit inserted at bit `position` (zero-based): the bits from
`position` up move one place higher.
"""
@inline function _insert_zero(t::Int, position::Int)
    low = t & ((1 << position) - 1)
    ((t ⊻ low) << 1) | low
end

"""
    _swap_pair(t, i, j) -> (b0, b1)

The one-based indices of the `t`-th pair (zero-based) of assignments that
differ by exchanging qubits `i < j` with different bits: `b0` has qubit `i`
at 0 and `j` at 1, `b1` the opposite, and both agree in all other qubits.
"""
@inline function _swap_pair(t::Int, i::Int, j::Int)
    base = _insert_zero(_insert_zero(t, i - 1), j - 1)
    (base | (1 << (j - 1))) + 1, (base | (1 << (i - 1))) + 1
end

"""
    _swap_run!(ψ, i, j, t0, t1, c, s)

The XY mixer on qubits `i < j`: turn the pairs of [`_swap_pair`](@ref)
`t ∈ t0:t1`, which is `exp(-iθ(XᵢXⱼ + YᵢYⱼ)/2)`. The other amplitudes, with
equal bits on `i` and `j`, do not change.
"""
@inline function _swap_run!(ψ::Vector{ComplexF64}, i::Int, j::Int, t0::Int, t1::Int, c::Float64, s::Float64)
    @inbounds for t in t0:t1
        b0, b1 = _swap_pair(t, i, j)
        ψ[b0], ψ[b1] = _turn(ψ[b0], ψ[b1], c, s)
    end
end

"""
    _swap_run_adjoint!(ψ, λ, i, j, t0, t1, c, s) -> Σ Im⟨λ|G|ψ⟩

Undo [`_swap_run!`](@ref) on `ψ` and `λ` and return this range's part of
`Im⟨λ|G|ψ⟩` for `G = (XᵢXⱼ + YᵢYⱼ)/2`.
"""
@inline function _swap_run_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, i::Int, j::Int,
                                    t0::Int, t1::Int, c::Float64, s::Float64)
    g = 0.0
    @inbounds for t in t0:t1
        b0, b1 = _swap_pair(t, i, j)
        a0 = ψ[b0]; a1 = ψ[b1]; l0 = λ[b0]; l1 = λ[b1]
        g += _imdot(l0, a1) + _imdot(l1, a0)
        ψ[b0], ψ[b1] = _turn(a0, a1, c, s)
        λ[b0], λ[b1] = _turn(l0, l1, c, s)
    end
    g
end

# --- the low qubits, inside one block -----------------------------------------------

"""
    _low_qubits!(ψ, base, nbits, c, s)

Turn qubits `1:nbits` inside the block `ψ[base+1 : base+2^nbits]`, qubit `q`
by the angle with cosine `c[q]` and sine `s[q]`.
"""
@inline function _low_qubits!(ψ::Vector{ComplexF64}, base::Int, nbits::Int,
                              c::Vector{Float64}, s::Vector{Float64})
    nbits ≥ 1 || return
    c1 = c[1]; s1 = s[1]
    @inbounds for b in base+1:2:base+(1<<nbits)
        ψ[b], ψ[b+1] = _turn(ψ[b], ψ[b+1], c1, s1)
    end
    for q in 2:nbits
        stride = 1 << (q - 1)
        for group in base:2stride:base+(1<<nbits)-1
            _pair_run!(ψ, stride, group + 1, group + stride, c[q], s[q])
        end
    end
end

"""
    _low_qubits_adjoint!(ψ, λ, base, nbits, c, s, w) -> g

Undo [`_low_qubits!`](@ref) on one block of `ψ` and `λ` (`s` negated) and
return `Σ_q w[q] Im⟨λ|X_q|ψ⟩` over this block.
"""
@inline function _low_qubits_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, base::Int,
                                      nbits::Int, c::Vector{Float64}, s::Vector{Float64},
                                      w::Vector{Float64})
    g = 0.0
    for q in 1:nbits
        stride = 1 << (q - 1)
        gq = 0.0
        for group in base:2stride:base+(1<<nbits)-1
            gq += _pair_run_adjoint!(ψ, λ, stride, group + 1, group + stride, c[q], s[q])
        end
        g += w[q] * gq
    end
    g
end
