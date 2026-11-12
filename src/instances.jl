# instances.jl — common problems, ready to solve.

"""
    max_cut(n, edges; weights = nothing) -> IsingProblem

Wrapper function setting up an instance of the MaxCut problem for a graph with the nodes `1:n`.

### Input
- `n::Integer`: The number of nodes.
- `edges`: The edges of the graph, as pairs of nodes `(i, j)`.
- `weights` (optional): One weight ``w_{ij}`` per edge, 1 by default.

### Output
- An `IsingProblem` to be maximized.

### Notes
Spin ``\\sigma_i = \\pm 1`` says which of the two groups node ``i`` joins, and the cost function is the weight of the cut,
```math
C(\\sigma) = \\sum_{(i, j) \\in E(G)} w_{ij} \\frac{1 - \\sigma_i \\sigma_j}{2},
```
where ``E(G)`` is the set of edges of the graph ``G``. With `Graphs.jl` loaded, `max_cut(g)` takes a graph `g` instead.
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

Wrapper function setting up an instance of the minimum vertex cover problem for a graph with the nodes `1:n`.

### Input
- `n::Integer`: The number of nodes.
- `edges`: The edges of the graph, as pairs of nodes `(i, j)`.
- `penalty::Real=2` (optional): The penalty ``P > 1`` for each edge without a chosen end.

### Output
- A `QUBOProblem` to be minimized.

### Notes
Bit ``x_i = 1`` means that node ``i`` is chosen. The cost function
```math
C(x) = \\sum_{i \\in V(G)} x_i + P \\sum_{(i, j) \\in E(G)} (1 - x_i)(1 - x_j)
```
counts the chosen nodes plus ``P`` for each edge without a chosen end, where ``V(G)`` is the set of vertices and ``E(G)`` is the set of edges of the graph ``G``. With ``P > 1``, every minimum is a cover: adding a node to cover an edge costs 1 and saves at least ``P``. With `Graphs.jl` loaded, `min_vertex_cover(g)` takes a graph `g` instead.
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

Wrapper function setting up an instance of the Sherrington-Kirkpatrick model.

### Input
- `n::Integer`: The number of spins.
- `seed::Integer=1` (optional): The seed for the random-number generator `Xoshiro` used for the couplings.

### Output
- An `IsingProblem` to be minimized.

### Notes
The cost function is
```math
C(\\sigma) = \\frac{1}{\\sqrt{n}} \\sum_{i<j\\leq n} J_{ij} \\sigma_i \\sigma_j,
```
where the couplings ``J_{ij}`` are i.i.d. standard Gaussian variables. The problem stores ``J_{ij}/\\sqrt{n}`` as its couplings.
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

Wrapper function setting up an instance of the partition problem.

### Input
- `a::AbstractVector{<:Real}`: The numbers to be partitioned.

### Output
- An `IsingProblem` to be minimized.

### Notes
The partition problem for a set of numbers ``\\mathcal{S} = \\{a_1, ..., a_N\\}`` consists of finding two subsets ``\\mathcal{S}_{1} \\cup \\mathcal{S}_2 = \\mathcal{S}`` such that the difference of the sums over the two subsets is as small as possible. Spin ``\\sigma_i = \\pm 1`` says which subset ``a_i`` joins, and the cost function is the squared difference
```math
C(\\sigma) = \\Big(\\sum_{i=1}^{N} a_i \\sigma_i\\Big)^2 = \\sum_{i<j\\leq N} J_{ij} \\sigma_i \\sigma_j + \\sum_{i=1}^N a_i^2
```
with ``J_{ij} = 2 a_i a_j``.
"""
function partition_problem(a::AbstractVector{<:Real})
    J = 2 .* (a * a')
    J[diagind(J)] .= 0
    IsingProblem(zeros(length(a)), J; offset = sum(abs2, a), sense = :min)
end
