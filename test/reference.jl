# Independent reference simulators for the tests: nothing here uses QAOA.jl's kernels.
#
# Conventions, as in QAOA.jl: amplitude b + 1 belongs to the assignment whose qubit i is
# bit i - 1 of b; for Ising problems bit 1 means σ = +1. Layer k applies
# exp(-iγ[k] f) with f = ±(C - offset)/s (+ for a maximization), then the mixer.

"""
    spins(b, n) -> Vector{Int}

The spins `σᵢ = 2bᵢ - 1` of assignment `b`.
"""
spins(b::Integer, n::Integer) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:n]

"""
    bits(b, n) -> Vector{Int}

The bits `xᵢ` of assignment `b`.
"""
bits(b::Integer, n::Integer) = [(b >> (i - 1)) & 1 for i in 1:n]

"""
    objectives(prob) -> Vector{Float64}

The objective of every assignment, by brute force with `objective`.
"""
function objectives(prob)
    n = QAOA.nvariables(prob)
    [prob isa IsingProblem ? objective(prob, spins(b, n)) : objective(prob, bits(b, n)) for b in 0:2^n-1]
end

"""
    phase_vector(prob) -> Vector{Float64}

`f = ±(C - offset)/s` for every assignment.
"""
phase_vector(prob) = (prob.sense === :max ? 1 : -1) .* (objectives(prob) .- QAOA._offset(prob)) ./ fieldscale(prob)

"""
    pauli(k, P, n) -> Matrix

The single-qubit matrix `P` on qubit `k` of `n` (qubit 1 is the least significant bit).
"""
pauli(k::Integer, P::AbstractMatrix, n::Integer) = reduce(kron, [i == k ? P : Matrix(1.0I, 2, 2) for i in n:-1:1])

const PX = ComplexF64[0 1; 1 0]
const PY = ComplexF64[0 -im; im 0]

"""
    exact_state(prob, sched; Δ = ones(n), pairs = nothing, ψ0 = nothing) -> Vector{ComplexF64}

The QAOA state from dense matrix exponentials: the X mixer `exp(-iβ Σ Δᵢ Xᵢ)`, or,
with `pairs`, the product of `exp(-iβ (XᵢXⱼ + YᵢYⱼ)/2)` over the pairs.
"""
function exact_state(prob, sched; Δ = ones(QAOA.nvariables(prob)), pairs = nothing, ψ0 = nothing)
    n = QAOA.nvariables(prob)
    f = phase_vector(prob)
    ψ = ψ0 === nothing ? fill(ComplexF64(2.0^(-n / 2)), 2^n) : ComplexF64.(ψ0) ./ norm(ψ0)
    B = sum(Δ[k] * pauli(k, PX, n) for k in 1:n)
    for k in 1:length(sched)
        ψ = cis.(-sched.γ[k] .* f) .* ψ
        if pairs === nothing
            ψ = exp(-im * sched.β[k] * B) * ψ
        else
            for (i, j) in pairs
                G = (pauli(i, PX, n) * pauli(j, PX, n) + pauli(i, PY, n) * pauli(j, PY, n)) / 2
                ψ = exp(-im * sched.β[k] * G) * ψ
            end
        end
    end
    ψ
end

"""
    loop_state(prob, sched) -> Vector{ComplexF64}

The QAOA state with the X mixer from plain loops, one qubit at a time: fast enough
for the qubit counts at which QAOA.jl switches to blocked and paired passes.
"""
function loop_state(prob, sched)
    n = QAOA.nvariables(prob)
    f = phase_vector(prob)
    ψ = fill(ComplexF64(2.0^(-n / 2)), 2^n)
    for k in 1:length(sched)
        ψ .*= cis.(-sched.γ[k] .* f)
        c, s = cos(sched.β[k]), sin(sched.β[k])
        for q in 1:n, b in 0:2^n-1
            stride = 1 << (q - 1)
            b & stride == 0 || continue
            a0, a1 = ψ[b+1], ψ[b+1+stride]
            ψ[b+1], ψ[b+1+stride] = c * a0 - im * s * a1, c * a1 - im * s * a0
        end
    end
    ψ
end

"""
    finite_difference_gradient(ws, sched; ε = 1e-6) -> (∂β, ∂γ)

Central differences of `expectation` with respect to every angle.
"""
function finite_difference_gradient(ws, sched; ε = 1e-6)
    p = length(sched)
    angles = vcat(sched.β, sched.γ)
    g = map(1:2p) do k
        up, down = copy(angles), copy(angles)
        up[k] += ε
        down[k] -= ε
        (expectation(ws, Schedule(up[1:p], up[p+1:end])) -
         expectation(ws, Schedule(down[1:p], down[p+1:end]))) / 2ε
    end
    g[1:p], g[p+1:end]
end

"""
    random_ising(n; fields = true, seed = 1, kwargs...) -> IsingProblem

A dense random Ising problem with Gaussian fields and couplings.
"""
function random_ising(n; fields = true, seed = 1, kwargs...)
    rng = Xoshiro(seed)
    J = randn(rng, n, n)
    J = (J + J') / 2
    J[diagind(J)] .= 0
    IsingProblem(fields ? randn(rng, n) : zeros(n), J; kwargs...)
end

"""
    gnp_edges(n, probability, seed) -> Vector{Tuple{Int,Int}}

The edges of a random graph G(n, probability).
"""
function gnp_edges(n, probability, seed)
    rng = Xoshiro(seed)
    [(i, j) for j in 1:n for i in 1:j-1 if rand(rng) < probability]
end
