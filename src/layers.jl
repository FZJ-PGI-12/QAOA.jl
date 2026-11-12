# layers.jl — one QAOA layer, forward and backward, as passes over the state.
#
# With the X mixer, a layer takes few passes over memory: one pass over blocks of
# 2^BLOCK_BITS amplitudes applies the phase separator and turns the low qubits, then
# the remaining qubits are turned two per pass, and a folded state ends with the
# folded qubit. The backward layer undoes the same passes in reverse order.

# --- whole-state passes -------------------------------------------------------------

"""
    _blocked_pass!(ψ, d, γ, nbits, c, s, threaded)

Apply the phase separator with angle `γ` and turn qubits `1:nbits` (cosines
`c`, sines `s`), block by block. Each block of `2^nbits` amplitudes stays in
the cache for all of this.
"""
function _blocked_pass!(ψ::Vector{ComplexF64}, d, γ::Float64, nbits::Int,
                        c::Vector{Float64}, s::Vector{Float64}, threaded::Bool)
    B = 1 << nbits
    _chunks(length(ψ) >> nbits, threaded) do lo, hi
        for base in (lo-1)*B:B:(hi-1)*B
            _phase!(ψ, d, γ, base + 1, base + B)
            _low_qubits!(ψ, base, nbits, c, s)
        end
    end
end

"""
    _blocked_pass_adjoint!(ψ, λ, d, γ, nbits, c, s, w, threaded) -> (gβ, gγ)

Undo [`_blocked_pass!`](@ref) on `ψ` and `λ`, block by block: first the turns
of qubits `1:nbits` (`s` negated), then the phase separator. Return
`Σ_q w[q] Im⟨λ|X_q|ψ⟩` over these qubits and `Im⟨λ|f|ψ⟩`.
"""
function _blocked_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, d, γ::Float64,
                                nbits::Int, c::Vector{Float64}, s::Vector{Float64},
                                w::Vector{Float64}, threaded::Bool)
    B = 1 << nbits
    _chunked_sum(length(ψ) >> nbits, threaded, (0.0, 0.0)) do lo, hi
        gβ = gγ = 0.0
        for base in (lo-1)*B:B:(hi-1)*B
            gβ += _low_qubits_adjoint!(ψ, λ, base, nbits, c, s, w)
            gγ += _phase_adjoint!(ψ, λ, d, γ, base + 1, base + B)
        end
        (gβ, gγ)
    end
end

"""
    _high_passes!(ψ, first, last, c, s, threaded)

Turn qubits `first:last`, two per pass over memory (one in the last pass if
their number is odd).
"""
function _high_passes!(ψ::Vector{ComplexF64}, first::Int, last::Int,
                       c::Vector{Float64}, s::Vector{Float64}, threaded::Bool)
    q = first
    while q ≤ last
        if q < last
            _quad_pass!(ψ, q, c, s, threaded)
            q += 2
        else
            _pair_pass!(ψ, q, c, s, threaded)
            q += 1
        end
    end
end

"""
    _high_passes_adjoint!(ψ, λ, first, last, c, s, w, threaded) -> g

Undo [`_high_passes!`](@ref) on `ψ` and `λ` (`s` negated) and return
`Σ_q w[q] Im⟨λ|X_q|ψ⟩` over qubits `first:last`. The turns commute, so the
order does not matter.
"""
function _high_passes_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, first::Int, last::Int,
                               c::Vector{Float64}, s::Vector{Float64}, w::Vector{Float64},
                               threaded::Bool)
    g = 0.0
    q = first
    while q ≤ last
        if q < last
            g += _quad_pass_adjoint!(ψ, λ, q, c, s, w, threaded)
            q += 2
        else
            g += w[q] * _pair_pass_adjoint!(ψ, λ, q, c, s, threaded)
            q += 1
        end
    end
    g
end

"""
    _pair_pass!(ψ, q, c, s, threaded)

Turn qubit `q` of the whole state.
"""
function _pair_pass!(ψ::Vector{ComplexF64}, q::Int, c::Vector{Float64}, s::Vector{Float64}, threaded::Bool)
    stride = 1 << (q - 1)
    cq, sq = c[q], s[q]
    _chunks(length(ψ) ÷ 2, threaded) do lo, hi
        _runs((a, b) -> _pair_run!(ψ, stride, a, b, cq, sq), stride, 2stride, lo - 1, hi - 1, 0.0)
    end
end

"""
    _pair_pass_adjoint!(ψ, λ, q, c, s, threaded) -> Im⟨λ|X_q|ψ⟩

Undo the turn of qubit `q` on `ψ` and `λ` (`s` negated).
"""
function _pair_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, q::Int,
                             c::Vector{Float64}, s::Vector{Float64}, threaded::Bool)
    stride = 1 << (q - 1)
    cq, sq = c[q], s[q]
    _chunked_sum(length(ψ) ÷ 2, threaded, 0.0) do lo, hi
        _runs((a, b) -> _pair_run_adjoint!(ψ, λ, stride, a, b, cq, sq), stride, 2stride, lo - 1, hi - 1, 0.0)
    end
end

"""
    _quad_pass!(ψ, q, c, s, threaded)

Turn qubits `q` and `q + 1` of the whole state in one pass.
"""
function _quad_pass!(ψ::Vector{ComplexF64}, q::Int, c::Vector{Float64}, s::Vector{Float64}, threaded::Bool)
    stride = 1 << (q - 1)
    c1, s1, c2, s2 = c[q], s[q], c[q+1], s[q+1]
    _chunks(length(ψ) ÷ 4, threaded) do lo, hi
        _runs((a, b) -> _quad_run!(ψ, stride, a, b, c1, s1, c2, s2), stride, 4stride, lo - 1, hi - 1, 0.0)
    end
end

"""
    _quad_pass_adjoint!(ψ, λ, q, c, s, w, threaded) -> g

Undo the turns of qubits `q` and `q + 1` on `ψ` and `λ` (`s` negated) and
return `w[q] Im⟨λ|X_q|ψ⟩ + w[q+1] Im⟨λ|X_{q+1}|ψ⟩`.
"""
function _quad_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, q::Int,
                             c::Vector{Float64}, s::Vector{Float64}, w::Vector{Float64}, threaded::Bool)
    stride = 1 << (q - 1)
    c1, s1, c2, s2, w1, w2 = c[q], s[q], c[q+1], s[q+1], w[q], w[q+1]
    _chunked_sum(length(ψ) ÷ 4, threaded, 0.0) do lo, hi
        _runs((a, b) -> _quad_run_adjoint!(ψ, λ, stride, a, b, c1, s1, c2, s2, w1, w2),
              stride, 4stride, lo - 1, hi - 1, 0.0)
    end
end

"""
    _fold_pass!(ψ, c, s, threaded)

Turn the folded qubit of a folded state; see [`_fold_run!`](@ref).
"""
function _fold_pass!(ψ::Vector{ComplexF64}, c::Float64, s::Float64, threaded::Bool)
    _chunks(length(ψ) ÷ 2, threaded) do lo, hi
        _fold_run!(ψ, lo, hi, c, s)
    end
end

"""
    _fold_pass_adjoint!(ψ, λ, c, s, threaded) -> Im⟨λ|X|ψ⟩

Undo the turn of the folded qubit on `ψ` and `λ` (`s` negated).
"""
function _fold_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, c::Float64, s::Float64,
                             threaded::Bool)
    _chunked_sum(length(ψ) ÷ 2, threaded, 0.0) do lo, hi
        _fold_run_adjoint!(ψ, λ, lo, hi, c, s)
    end
end

"""
    _phase_pass!(ψ, d, γ, threaded)

Apply the phase separator with angle `γ` to the whole state.
"""
function _phase_pass!(ψ::Vector{ComplexF64}, d, γ::Float64, threaded::Bool)
    _chunks(length(ψ), threaded) do lo, hi
        _phase!(ψ, d, γ, lo, hi)
    end
end

"""
    _phase_pass_adjoint!(ψ, λ, d, γ, threaded) -> Im⟨λ|f|ψ⟩

Undo the phase separator on `ψ` and `λ`.
"""
function _phase_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, d, γ::Float64, threaded::Bool)
    _chunked_sum(length(ψ), threaded, 0.0) do lo, hi
        _phase_adjoint!(ψ, λ, d, γ, lo, hi)
    end
end

"""
    _swap_pass!(ψ, i, j, c, s, threaded)

Apply the XY mixer of the pair `(i, j)` to the whole state.
"""
function _swap_pass!(ψ::Vector{ComplexF64}, i::Int, j::Int, c::Float64, s::Float64, threaded::Bool)
    _chunks(length(ψ) ÷ 4, threaded) do lo, hi
        _swap_run!(ψ, i, j, lo - 1, hi - 1, c, s)
    end
end

"""
    _swap_pass_adjoint!(ψ, λ, i, j, c, s, threaded) -> Im⟨λ|G|ψ⟩

Undo the XY mixer of the pair `(i, j)` on `ψ` and `λ` (`s` negated).
"""
function _swap_pass_adjoint!(ψ::Vector{ComplexF64}, λ::Vector{ComplexF64}, i::Int, j::Int,
                             c::Float64, s::Float64, threaded::Bool)
    _chunked_sum(length(ψ) ÷ 4, threaded, 0.0) do lo, hi
        _swap_run_adjoint!(ψ, λ, i, j, lo - 1, hi - 1, c, s)
    end
end

# --- layers ---------------------------------------------------------------------------

"""
    _layer!(ws, γ, β)

Apply one QAOA layer to `ws.state`: the phase separator with angle `γ`, then
the mixer with angle `β`.
"""
function _layer!(ws::Workspace{<:Any,<:Any,XMixer}, γ::Float64, β::Float64)
    _set_turns!(ws, β)
    _prepare!(ws.diagonal, γ)
    ψ, c, s, threaded = ws.state, ws.cosines, ws.sines, _threaded(ws)
    m = _stored_qubits(ws)
    nbits = min(BLOCK_BITS, m)
    _blocked_pass!(ψ, ws.diagonal, γ, nbits, c, s, threaded)
    _high_passes!(ψ, nbits + 1, m, c, s, threaded)
    ws.folded && _fold_pass!(ψ, c[end], s[end], threaded)
    nothing
end

function _layer!(ws::Workspace{<:Any,<:Any,XYMixer}, γ::Float64, β::Float64)
    _prepare!(ws.diagonal, γ)
    threaded = _threaded(ws)
    _phase_pass!(ws.state, ws.diagonal, γ, threaded)
    s, c = sincos(β)
    for (i, j) in ws.mixer.pairs
        _swap_pass!(ws.state, i, j, c, s, threaded)
    end
    nothing
end

"""
    _layer_adjoint!(ws, γ, β) -> (gβ, gγ)

Undo one layer on the state `ws.state` and the adjoint vector `ws.adjoint`,
in reverse order, and return the halves of the derivatives of `⟨f⟩` with
respect to the layer's `β` and `γ`: `Σ_q Δ[q] Im⟨λ|X_q|ψ⟩` (or the XY mixer's
sum) and `Im⟨λ|f|ψ⟩`.
"""
function _layer_adjoint!(ws::Workspace{<:Any,<:Any,XMixer}, γ::Float64, β::Float64)
    _set_turns!(ws, -β)
    _prepare!(ws.diagonal, -γ)
    ψ, λ, c, s, w = ws.state, ws.adjoint, ws.cosines, ws.sines, ws.factors
    threaded = _threaded(ws)
    m = _stored_qubits(ws)
    nbits = min(BLOCK_BITS, m)
    gβ = ws.folded ? w[end] * _fold_pass_adjoint!(ψ, λ, c[end], s[end], threaded) : 0.0
    gβ += _high_passes_adjoint!(ψ, λ, nbits + 1, m, c, s, w, threaded)
    gβ_low, gγ = _blocked_pass_adjoint!(ψ, λ, ws.diagonal, γ, nbits, c, s, w, threaded)
    gβ + gβ_low, gγ
end

function _layer_adjoint!(ws::Workspace{<:Any,<:Any,XYMixer}, γ::Float64, β::Float64)
    ψ, λ, threaded = ws.state, ws.adjoint, _threaded(ws)
    s, c = sincos(-β)
    gβ = 0.0
    for (i, j) in Iterators.reverse(ws.mixer.pairs)
        gβ += _swap_pass_adjoint!(ψ, λ, i, j, c, s, threaded)
    end
    _prepare!(ws.diagonal, -γ)
    gβ, _phase_pass_adjoint!(ψ, λ, ws.diagonal, γ, threaded)
end
