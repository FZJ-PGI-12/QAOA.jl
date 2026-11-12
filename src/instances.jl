# instances.jl — common problems, ready to solve.

"""
    max_cut(n, edges; weights = nothing) -> IsingProblem

The MaxCut problem on a graph with nodes `1:n`: split the nodes into two
groups so that the edges between the groups have the largest total weight.
Spin `σ[i] = ±1` says which group node `i` joins, and the objective

    Σ_{(i,j) ∈ edges} w_ij (1 - σ[i] σ[j]) / 2

is the weight of the cut, to be maximized. `edges` is a collection of node
pairs `(i, j)`; `weights` gives one weight per edge, 1 by default. With
Graphs.jl loaded, `max_cut(g)` takes a graph.
"""
function max_cut(n::Integer, edges; weights = nothing)
    J = zeros(n, n)
    total = 0.0
    for (k, (i, j)) in enumerate(edges)
        i == j && throw(ArgumentError("edge $k joins node $i to itself"))
        w = weights === nothing ? 1.0 : Float64(weights[k])
        J[i, j] -= w / 2
        J[j, i] -= w / 2
        total += w
    end
    IsingProblem(zeros(n), J; offset = total / 2, sense = :max)
end

"""
    min_vertex_cover(n, edges; penalty = 2) -> QUBOProblem

The minimum vertex cover of a graph with nodes `1:n`: choose as few nodes as
possible (`x[i] = 1`) such that every edge has at least one chosen end. The
objective

    Σᵢ x[i] + penalty Σ_{(i,j) ∈ edges} (1 - x[i]) (1 - x[j])

counts the chosen nodes plus `penalty` for each edge without a chosen end.
With `penalty > 1`, every minimum is a cover: adding a node to cover an edge
costs 1 and saves at least `penalty`.
"""
function min_vertex_cover(n::Integer, edges; penalty::Real = 2)
    penalty > 1 || throw(ArgumentError("penalty must exceed 1, got $penalty"))
    Q = zeros(n, n)
    c = ones(n)
    constant = 0.0
    for (i, j) in edges
        i == j && throw(ArgumentError("an edge joins node $i to itself"))
        Q[min(i, j), max(i, j)] += penalty            # (1 - x[i])(1 - x[j]) = 1 - x[i] - x[j] + x[i] x[j]
        c[i] -= penalty
        c[j] -= penalty
        constant += penalty
    end
    QUBOProblem(Q; c, constant, sense = :min)
end

"""
    sherrington_kirkpatrick(n; seed = 1) -> IsingProblem

An instance of the Sherrington–Kirkpatrick spin glass: minimize the energy

    Σ_{i<j} J[i,j] σ[i] σ[j] / √n

with couplings `J[i,j]` drawn independently from the standard normal
distribution, using the random-number generator `Xoshiro(seed)`.
"""
function sherrington_kirkpatrick(n::Integer; seed::Integer = 1)
    rng = Xoshiro(seed)
    J = zeros(n, n)
    for j in 1:n, i in 1:j-1
        J[i, j] = J[j, i] = randn(rng) / sqrt(n)
    end
    IsingProblem(zeros(n), J; sense = :min)
end

"""
    partition_problem(a) -> IsingProblem

The number partitioning problem: split the numbers `a` into two groups whose
sums are as close as possible. Spin `σ[i] = ±1` says which group `a[i]`
joins, and the objective is the squared difference of the two sums,

    (Σᵢ a[i] σ[i])² = Σᵢ a[i]² + Σ_{i<j} 2 a[i] a[j] σ[i] σ[j],

to be minimized.
"""
function partition_problem(a::AbstractVector{<:Real})
    J = 2 .* (a * a')
    J[diagind(J)] .= 0
    IsingProblem(zeros(length(a)), J; offset = sum(abs2, a), sense = :min)
end
