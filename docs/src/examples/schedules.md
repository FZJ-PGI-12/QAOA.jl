# [Schedules and annealing](@id Schedules)

A [`Schedule`](@ref) holds the angles `β[k]` and `γ[k]` of the `p` layers of
the QAOA. This page shows three ways to choose them: annealing, a schedule of
your own, and optimization. All examples use MaxCut on the Petersen graph,
whose largest cut has 12 edges:

```@example schedules
using QAOA
petersen_edges = [(1, 2), (1, 5), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8), (4, 5), (4, 9),
                  (5, 10), (6, 8), (6, 9), (7, 9), (7, 10), (8, 10)]
problem = max_cut(10, petersen_edges)
ws = Workspace(problem)
spins(b) = [2 * ((b >> (i - 1)) & 1) - 1 for i in 1:10]
largest = [objective(problem, spins(b)) == 12 for b in 0:2^10-1]
probability_of_largest_cut(schedule) = sum(probabilities(ws, schedule)[largest])
nothing # hide
```

## Annealing

A quantum annealing turns the mixer slowly into the objective. Cut into `p`
steps of length `τ`, it is a QAOA with the angles of
[`linear_schedule(p, τ)`](@ref linear_schedule):

```math
\gamma_k = \tau \frac{k}{p}, \qquad \beta_k = \tau \Big(1 - \frac{k - 1}{p}\Big).
```

The total annealing time is `pτ`. Since `γ` is measured in units of the
problem's [`fieldscale`](@ref), the same `τ` suits many problems; values
between 0.5 and 0.7 usually work well. Longer annealing finds the largest cut
more often:

```@example schedules
[probability_of_largest_cut(linear_schedule(p)) for p in (10, 20, 40, 80)]
```

## A schedule of your own

Any two vectors of angles make a schedule. An annealing path `s(t)` from 0 to
1 gives `γ[k] = τ s(tₖ)` and `β[k] = τ (1 - s(tₖ))`. A path that starts and
ends slowly, `s(t) = sin²(πt/2)`:

```@example schedules
function sine_schedule(p, τ = 0.5)
    s = [sin(π * k / 2p)^2 for k in 1:p]
    Schedule(τ .* (1 .- s), τ .* s)
end
[probability_of_largest_cut(sine_schedule(p)) for p in (10, 20, 40, 80)]
```

For this problem, the linear path is better with few layers.

## Optimization

[`optimize_schedule`](@ref) tunes the angles with the BFGS method, using exact
derivatives. Started from a schedule, it finds a nearby optimum:

```@example schedules
tuned, value = optimize_schedule(ws, linear_schedule(3))
value
```

Given a number of layers instead, it builds the schedule layer by layer: it
optimizes one layer, and then starts each further layer from the
[`interpolate_schedule`](@ref) of the previous optimum. Good angles change
smoothly with the number of layers, so each search starts close to its
optimum.

```@example schedules
tuned, value = optimize_schedule(ws, 4)
probability_of_largest_cut(tuned)
```

## Other optimizers

[`expectation_and_gradient!`](@ref) returns the expectation value and writes
its derivatives with respect to `β` and `γ` into two vectors. A
gradient-based optimizer needs nothing more. With [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
for example:

```julia
using Optim

# the negative average cut and its gradient: Optim minimizes, the cut is maximized
function negative_cut!(F, G, angles)
    p = length(angles) ÷ 2
    schedule = Schedule(angles[1:p], angles[p+1:end])
    if G === nothing
        value = expectation(ws, schedule)
    else
        value = expectation_and_gradient!(view(G, 1:p), view(G, p+1:2p), ws, schedule)
        G .*= -1
    end
    F === nothing ? nothing : -value
end

start = linear_schedule(4)
result = optimize(Optim.NLSolversBase.only_fg!(negative_cut!), vcat(start.β, start.γ), LBFGS())
-Optim.minimum(result)      # 11.9365…, the value optimize_schedule finds from the same start
```
