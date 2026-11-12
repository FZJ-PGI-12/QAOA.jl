"""
    QAOAYaoExt

The QAOA as a Yao.jl circuit, loaded with `using Yao`: for running it on other
simulators or hardware, or exporting it.
"""
module QAOAYaoExt

using QAOA
using QAOA: IsingProblem, QUBOProblem, Schedule, XMixer, XYMixer, nvariables, fieldscale
using Yao: chain, put, repeat, H, Rx, Rz, rot, kron, Z, X, Y, matblock, AbstractBlock

"""
    circuit(prob, sched; mixer = XMixer()) -> Yao block

The QAOA for `prob` with the angles `sched` as a Yao.jl circuit on
`nvariables(prob)` qubits, gate for gate the same algorithm as
[`evolve!`](@ref):

- With the X mixer, the circuit starts with a Hadamard gate on every qubit,
  which turns |0…0⟩ into the usual initial state. With an [`XYMixer`](@ref),
  apply the circuit to your own initial state, such as
  `ArrayReg(dicke_state(n, k))`.
- The phase separator of layer `k` is an `Rz` gate per field and an
  `exp(-iθ ZᵢZⱼ/2)` gate per nonzero coupling. Since a measured 1 means
  `σ = +1`, each spin is `σᵢ = -Zᵢ`.
- The mixer is an `Rx` gate per qubit, or one two-qubit gate per pair of the
  XY mixer.

A [`QUBOProblem`](@ref) is converted to its Ising form first; that changes
the state by a global phase only.
"""
function QAOA.circuit(prob::IsingProblem, sched::Schedule; mixer = XMixer())
    n = nvariables(prob)
    layers = [_layer(prob, sched.γ[k], sched.β[k], mixer) for k in 1:length(sched)]
    mixer isa XMixer ? chain(n, repeat(n, H), layers...) : chain(n, layers...)
end
QAOA.circuit(prob::QUBOProblem, sched::Schedule; mixer = XMixer()) =
    QAOA.circuit(IsingProblem(prob), sched; mixer)
QAOA.circuit(prob, sched::Schedule; mixer = XMixer()) =
    throw(ArgumentError("a circuit needs an IsingProblem or a QUBOProblem, got $(typeof(prob))"))

"""
    _layer(prob, γ, β, mixer) -> Yao block

One layer: `exp(-iγ f)` with `f = ±(C - offset)/s` in gates, then the mixer.
"""
_layer(prob::IsingProblem, γ, β, mixer) = chain(nvariables(prob), _phase_gates(prob, γ), _mixer_gates(prob, β, mixer))

"""
    _phase_gates(prob, γ) -> Yao block

The phase separator `exp(-iγ f)`. With `σᵢ = -Zᵢ` and `a = ±γ/s`, a field
`hᵢ σᵢ` gives `exp(+i a hᵢ Zᵢ) = Rz(-2a hᵢ)` and a coupling `Jᵢⱼ σᵢ σⱼ` gives
`exp(-i a Jᵢⱼ ZᵢZⱼ) = rot(ZZ, 2a Jᵢⱼ)`.
"""
function _phase_gates(prob::IsingProblem, γ)
    n = nvariables(prob)
    a = (prob.sense === :max ? 1 : -1) * γ / fieldscale(prob)
    fields = [put(n, i => Rz(-2a * prob.h[i])) for i in 1:n if !iszero(prob.h[i])]
    couplings = [put(n, (i, j) => rot(kron(Z, Z), 2a * prob.J[i, j]))
                 for j in 1:n for i in 1:j-1 if !iszero(prob.J[i, j])]
    chain(n, fields..., couplings...)
end

"""
    _mixer_gates(prob, β, mixer) -> Yao block

The mixer: `Rx(2βΔᵢ) = exp(-iβΔᵢXᵢ)` on every qubit, or `exp(-iβ(XᵢXⱼ + YᵢYⱼ)/2)`
on every pair of an XY mixer, in order.
"""
function _mixer_gates(prob::IsingProblem, β, ::XMixer)
    n = nvariables(prob)
    Δ = prob.Δ === nothing ? ones(n) : prob.Δ
    chain(n, [put(n, i => Rx(2β * Δ[i])) for i in 1:n]...)
end

function _mixer_gates(prob::IsingProblem, β, mixer::XYMixer)
    n = nvariables(prob)
    G = ComplexF64[0 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 0]    # (XX + YY)/2 on two qubits
    U = matblock(exp(-im * β * G))
    chain(n, [put(n, (i, j) => U) for (i, j) in mixer.pairs]...)
end

end
