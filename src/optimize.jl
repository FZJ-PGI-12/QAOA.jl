# optimize.jl — tuning the angles.

"""
    optimize_schedule(ws, sched; maxiter = 500, gtol = 1e-6) -> (Schedule, Float64)
    optimize_schedule(prob, sched; maxiter, gtol, kwargs...) -> (Schedule, Float64)

Angles near `sched` that optimize the expectation value of the objective
(minimize it for `sense = :min`, maximize it for `:max`), and that expectation
value.

The search is a local one: the BFGS method, starting from `sched`, with exact
derivatives from [`expectation_and_gradient!`](@ref). It stops when no
derivative exceeds `gtol` in units of the [`fieldscale`](@ref), or after
`maxiter` steps. The second form builds a [`Workspace`](@ref) with the
keywords `kwargs` first.
"""
function optimize_schedule(ws::Workspace, sched::Schedule; maxiter::Integer = 500, gtol::Real = 1e-6)
    p = length(sched)
    angles = _bfgs(_AngleObjective(ws, p), vcat(sched.β, sched.γ); maxiter, gtol)
    best = Schedule(angles[1:p], angles[p+1:2p])
    best, expectation(ws, best)
end
optimize_schedule(prob::AbstractProblem, sched::Schedule; maxiter::Integer = 500, gtol::Real = 1e-6, kwargs...) =
    optimize_schedule(Workspace(prob; kwargs...), sched; maxiter, gtol)

"""
    optimize_schedule(ws, p::Integer; τ = 0.5, maxiter = 500, gtol = 1e-6) -> (Schedule, Float64)
    optimize_schedule(prob, p::Integer; τ, maxiter, gtol, kwargs...) -> (Schedule, Float64)

Optimized angles for `p` layers, found layer by layer: optimize one layer,
starting from [`linear_schedule(1, τ)`](@ref linear_schedule); then, for each
further layer, start from the [`interpolate_schedule`](@ref) of the previous
optimum. Good angles usually change smoothly with the number of layers, so
each search starts close to its optimum.
"""
function optimize_schedule(ws::Workspace, p::Integer; τ::Real = 0.5, maxiter::Integer = 500, gtol::Real = 1e-6)
    p ≥ 1 || throw(ArgumentError("need at least one layer, got p = $p"))
    sched, value = optimize_schedule(ws, linear_schedule(1, τ); maxiter, gtol)
    for _ in 2:p
        sched, value = optimize_schedule(ws, interpolate_schedule(sched); maxiter, gtol)
    end
    sched, value
end
optimize_schedule(prob::AbstractProblem, p::Integer; τ::Real = 0.5, maxiter::Integer = 500,
                  gtol::Real = 1e-6, kwargs...) =
    optimize_schedule(Workspace(prob; kwargs...), p; τ, maxiter, gtol)

"""
    _AngleObjective(ws, p)

The function the angle search minimizes: `-⟨f⟩` as a function of the angles
`[β; γ]` of `p` layers, in units of the field scale and with the sign that
turns every problem into a minimization. Calling it as `obj(g, angles)`
returns the value and writes the gradient into `g`.
"""
struct _AngleObjective{W<:Workspace}
    ws::W
    p::Int
end

function (obj::_AngleObjective)(g::Vector{Float64}, angles::Vector{Float64})
    p, ws = obj.p, obj.ws
    sched = Schedule(angles[1:p], angles[p+1:2p])
    E = expectation_and_gradient!(view(g, 1:p), view(g, p+1:2p), ws, sched)
    g .*= -1 / (ws.sign * ws.scale)
    -(E - ws.offset) / (ws.sign * ws.scale)
end

"""
    _bfgs(objective, x0; maxiter, gtol) -> x

Minimize `objective` from `x0` with the BFGS method. `objective(g, x)` returns
the value at `x` and writes the gradient into `g`.

Steps are found by backtracking until the Armijo condition holds. The inverse
Hessian estimate starts as a multiple of the identity that limits the first
step to 0.1 in every coordinate: QAOA angles are periodic, and a long first
step lands in a distant, equivalent copy of the optimum. Stops when
`maximum(abs, g) ≤ gtol`, when a step no longer decreases the value by more
than rounding errors, or after `maxiter` steps.
"""
function _bfgs(objective, x0::Vector{Float64}; maxiter::Integer, gtol::Real)
    x, g = copy(x0), zeros(length(x0))
    value = objective(g, x)
    H = _initial_inverse_hessian(g)
    fresh = true
    xnew, gnew = similar(x), similar(g)
    for _ in 1:maxiter
        maximum(abs, g) ≤ gtol && break
        direction = -(H * g)
        slope = dot(g, direction)
        if slope ≥ 0                       # lost descent: start the estimate afresh
            H = _initial_inverse_hessian(g)
            fresh = true
            direction = -(H * g)
            slope = dot(g, direction)
        end
        step, newvalue = _backtrack!(xnew, gnew, objective, x, value, direction, slope)
        step == 0 && break
        _update_inverse_hessian!(H, xnew .- x, gnew .- g, fresh)
        fresh = false
        gain = value - newvalue
        x, xnew = xnew, x
        g, gnew = gnew, g
        value = newvalue
        gain ≤ 1e-14 * max(abs(value), 1) && break
    end
    x
end

"""
    _initial_inverse_hessian(g) -> Matrix

The identity, scaled so that the first step `-H g` moves no coordinate by more
than 0.1.
"""
_initial_inverse_hessian(g::Vector{Float64}) =
    Matrix{Float64}(I, length(g), length(g)) .* min(1.0, 0.1 / maximum(abs, g))

"""
    _backtrack!(xnew, gnew, objective, x, value, direction, slope) -> (step, newvalue)

Halve the step along `direction` from 1 until the Armijo condition
`objective(xnew) ≤ value + 1e-4 step slope` holds, writing the point and its
gradient into `xnew` and `gnew`. Returns step 0 if 40 halvings do not
suffice.
"""
function _backtrack!(xnew, gnew, objective, x, value, direction, slope)
    step = 1.0
    for _ in 1:40
        xnew .= x .+ step .* direction
        newvalue = objective(gnew, xnew)
        newvalue ≤ value + 1e-4 * step * slope && return step, newvalue
        step /= 2
    end
    0.0, value
end

"""
    _update_inverse_hessian!(H, s, y, first)

The BFGS update of the inverse Hessian estimate `H` with the step `s` and the
change of gradient `y`: `H ← (I - ρsyᵀ) H (I - ρysᵀ) + ρssᵀ` with
`ρ = 1/(yᵀs)`. Skipped unless `yᵀs > 0`. On the first step, `H` is first
scaled to `(yᵀs)/(yᵀy) I`.
"""
function _update_inverse_hessian!(H::Matrix{Float64}, s::Vector{Float64}, y::Vector{Float64}, first::Bool)
    sy = dot(s, y)
    sy > 1e-12 * norm(s) * norm(y) || return H
    first && (H .= (sy / dot(y, y)) .* Matrix{Float64}(I, size(H)...))
    ρ = 1 / sy
    Hy = H * y
    H .+= (ρ^2 * dot(y, Hy) + ρ) .* (s * s') .- ρ .* (s * Hy' .+ Hy * s')
    H
end
