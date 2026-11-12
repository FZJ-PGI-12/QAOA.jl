# workspace.jl — mixers, initial states, and the memory a QAOA run works in.

"""
    AbstractMixer

Supertype of the mixers [`XMixer`](@ref) and [`XYMixer`](@ref).
"""
abstract type AbstractMixer end

"""
    XMixer()

The standard mixer. In a layer with angle `β`, every qubit `i` turns by
`exp(-iβ Δ[i] X[i])`, where `Δ[i]` is the mixer factor of an
[`IsingProblem`](@ref) (1 by default).
"""
struct XMixer <: AbstractMixer end

"""
    XYMixer(pairs)

A mixer that keeps the number of ones in every assignment: in a layer with
angle `β`, it applies `exp(-iβ (XᵢXⱼ + YᵢYⱼ)/2)` to each pair `(i, j)` of
qubits, in the given order. On two qubits, this operator moves amplitude
between |01⟩ and |10⟩ and leaves |00⟩ and |11⟩ alone.

Use it for problems with the constraint that exactly `k` variables are 1,
starting from [`dicke_state(n, k)`](@ref dicke_state): every assignment the
QAOA reaches then satisfies the constraint. Pairs along a ring,
`[(1, 2), (2, 3), …, (n, 1)]`, or between all variables are common choices.
"""
struct XYMixer <: AbstractMixer
    pairs::Vector{Tuple{Int,Int}}
    function XYMixer(pairs)
        sorted = [(Int(min(i, j)), Int(max(i, j))) for (i, j) in pairs]
        all(p -> p[1] != p[2], sorted) || throw(ArgumentError("a pair must join two different qubits"))
        isempty(sorted) && throw(ArgumentError("the XY mixer needs at least one pair"))
        new(sorted)
    end
end

"""
    _check_mixer(mixer, n)

Throw an `ArgumentError` if `mixer` refers to a qubit outside `1:n`.
"""
_check_mixer(::XMixer, n::Int) = nothing
function _check_mixer(mixer::XYMixer, n::Int)
    all(p -> 1 ≤ p[1] && p[2] ≤ n, mixer.pairs) ||
        throw(ArgumentError("the XY mixer refers to a qubit outside 1:$n"))
    nothing
end

"""
    dicke_state(n, k) -> Vector{ComplexF64}

The equal superposition of all assignments of `n` qubits with exactly `k`
ones, the usual initial state for the [`XYMixer`](@ref).
"""
function dicke_state(n::Integer, k::Integer)
    0 ≤ k ≤ n || throw(ArgumentError("need 0 ≤ k ≤ n, got k = $k, n = $n"))
    ψ = [count_ones(b) == k ? one(ComplexF64) : zero(ComplexF64) for b in 0:(1<<n)-1]
    ψ ./ sqrt(binomial(n, k))
end

"""
    Workspace(prob; mixer = XMixer(), initial_state = nothing, symmetry = true,
              threaded = Threads.nthreads() > 1)

Everything a QAOA run on `prob` needs, computed or allocated once: the
objective of every assignment and the state vector. Reuse a workspace for many
schedules of the same problem, as an angle optimization does.

- `mixer`: [`XMixer()`](@ref XMixer) or an [`XYMixer`](@ref).
- `initial_state`: a vector of `2^n` amplitudes; by default every qubit starts
  in |+⟩.
- `symmetry`: if the objective does not change when all variables flip (an
  [`IsingProblem`](@ref) without fields, such as MaxCut), the QAOA state has
  the same symmetry, and the workspace stores only half of it. This halves
  memory and time without changing any result. It needs the X mixer and the
  default initial state; `symmetry = false` turns it off.
- `threaded`: split the work on states of `2^15` amplitudes or more over
  Julia's threads (start Julia with `--threads`).

The objective of every assignment takes 8 bytes, or 2 for objectives with
integer weights, and the state 16 bytes; a gradient needs another 16.
"""
mutable struct Workspace{P<:AbstractProblem,D<:Union{DenseDiagonal,TableDiagonal},M<:AbstractMixer}
    const problem::P
    const diagonal::D
    const mixer::M
    const folded::Bool
    const threaded::Bool
    const sign::Float64                     # +1 for a maximization, -1 for a minimization
    const scale::Float64                    # fieldscale(problem)
    const offset::Float64                   # objective = sign * scale * f + offset
    const factors::Vector{Float64}          # mixer factor Δ[i] of every qubit
    const initial::Union{Nothing,Vector{ComplexF64}}
    const state::Vector{ComplexF64}
    adjoint::Vector{ComplexF64}             # allocated by the first gradient
    const cosines::Vector{Float64}          # cos(β Δ[i]) of the layer at hand
    const sines::Vector{Float64}            # sin(β Δ[i]) of the layer at hand
end

function Workspace(prob::AbstractProblem; mixer::AbstractMixer = XMixer(), initial_state = nothing,
                   symmetry::Bool = true, threaded::Bool = nthreads() > 1)
    n = nvariables(prob)
    _check_mixer(mixer, n)
    initial = _initial_state(initial_state, n)
    folded = symmetry && mixer isa XMixer && initial === nothing && n ≥ 2 && _flipsymmetric(prob)
    _check_memory(n - folded)
    diagonal = _diagonal(prob, folded)
    Δ = _factors(prob)
    factors = Δ === nothing ? ones(n) : Vector{Float64}(Δ)
    Workspace(prob, diagonal, mixer, folded, threaded, _sense_sign(prob), Float64(fieldscale(prob)),
              Float64(_offset(prob)), factors, initial, Vector{ComplexF64}(undef, length(diagonal)),
              ComplexF64[], zeros(n), zeros(n))
end

"""
    _initial_state(state, n) -> Union{Nothing, Vector{ComplexF64}}

`nothing` for the default initial state |+⟩ of every qubit; otherwise `state`
normalized, as a `Vector{ComplexF64}`, after checking its length `2^n`.
"""
_initial_state(::Nothing, n::Int) = nothing
function _initial_state(state::AbstractVector, n::Int)
    length(state) == 1 << n ||
        throw(ArgumentError("the initial state needs 2^$n = $(1 << n) amplitudes, got $(length(state))"))
    ψ = Vector{ComplexF64}(state)
    norm(ψ) > 0 || throw(ArgumentError("the initial state is zero"))
    ψ ./ norm(ψ)
end

"""
    _check_memory(m)

Throw an `ArgumentError` if a state vector with `2^m` amplitudes and the
objective of every assignment do not fit into this computer's memory.
"""
function _check_memory(m::Int)
    m ≤ 50 || throw(ArgumentError("2^$m amplitudes are far beyond any memory"))
    needed = 24.0 * 2.0^m
    available = Float64(Sys.total_memory())
    needed ≤ available || throw(ArgumentError(
        "a QAOA on $m stored qubits needs about $(round(needed / 2^30; digits = 1)) GiB, " *
        "more than this computer's $(round(available / 2^30; digits = 1)) GiB"))
    nothing
end

"""
    _stored_qubits(ws) -> Int

The number of qubits in the stored state: all of them, or one fewer if the
state is folded.
"""
_stored_qubits(ws::Workspace) = trailing_zeros(length(ws.state))

"""
    _threaded(ws) -> Bool

Whether the passes over this workspace's state run on several threads.
"""
_threaded(ws::Workspace) = ws.threaded && length(ws.state) ≥ THREADING_THRESHOLD && nthreads() > 1

"""
    _initialize!(ws)

Put the initial state into `ws.state`: equal amplitudes, normalized in the
stored (possibly folded) representation, or the given initial state.
"""
function _initialize!(ws::Workspace)
    if ws.initial === nothing
        fill!(ws.state, 1 / sqrt(length(ws.state)))
    else
        copyto!(ws.state, ws.initial)
    end
    ws
end

"""
    _set_turns!(ws, β)

Store `cos(β Δ[i])` and `sin(β Δ[i])` of every qubit for the mixer of a layer.
A negative `β` undoes the turns.
"""
function _set_turns!(ws::Workspace, β::Float64)
    for i in eachindex(ws.factors)
        ws.sines[i], ws.cosines[i] = sincos(β * ws.factors[i])
    end
    ws
end

"""
    show(io, ws::Workspace)

Print the problem, the number of qubits, and whether the state is folded.
"""
function Base.show(io::IO, ws::Workspace)
    n = nvariables(ws.problem)
    print(io, "Workspace for ", ws.problem, ": $n qubits",
          ws.folded ? ", folded to $(n - 1) by symmetry" : "",
          ws.diagonal isa TableDiagonal ? ", objective as a table" : "")
end
