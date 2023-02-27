"""
    sherrington_kirkpatrick(variance::Float64; seed::Float64=1.0, num_layers::Int=1, driver=X)

Wrapper function setting up an instance of the Sherrington-Kirkpatrick model.

### Input
- `N::Int`: The number of spins of the problem.
- `variance::Float64`: The variance of the distribution of the coupling matrix.
- `seed::Float64=1.0`: The seed for the random-number generator used in the coupling matrix.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The cost function of the Sherrington-Kirkpatrick model is

``
\\hat H_P = \\frac{1}{\\sqrt{N}}\\sum_{i<j\\leq N} J_{ij} \\hat{Z}_i \\hat{Z}_j,
``

where the couplings ``J_{ij}`` are i.i.d. standard Gaussian variables, 
i.e. with zero mean ``\\langle J_{ij} \\rangle = 0`` and variance `` \\langle J_{ij}^2 \\rangle = J^2``.
"""
function sherrington_kirkpatrick(N::Int, variance::Float64; seed::Int=1, num_layers::Int=1, driver=X)
    Random.seed!(seed)
    J = rand(Distributions.Normal(0, variance), N, N) ./ sqrt(N) 
    J[diagind(J)] .= 0.0
    J = UpperTriangular(J)
    J = J + transpose(J)

    QAOA.Problem(num_layers, zeros(N), J)
end


"""
    partition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)

Wrapper function setting up an instance of the partition problem.

### Input
- `a::Vector{Float64}`: The input vector of numbers to be partitioned.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The partition problem for a set of uniformly distributed numbers ``\\mathcal{S} = \\{a_1, ..., a_N\\}`` 
consists of finding two subsets ``\\mathcal{S}_{1} \\cup \\mathcal{S}_2 =  \\mathcal{S}`` 
such that the difference of the sums over the two subsets ``\\mathcal{S}_{1, 2}`` is as small as possible. 
The cost function in Ising form can be defined as 

``
\\hat C = -\\left(\\sum_{i=1}^{N} a_i \\hat{Z}_i\\right)^2 = \\sum_{i<j\\leq N} J_{ij} \\hat{Z}_i \\hat{Z}_j + \\mathrm{const.}
``

with ``J_{ij}=-2a_i a_j``. The goal is then to _maximize_ ``\\hat C``.
"""
function partition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)
    J = -2 .* (a * transpose(a))
    J[diagind(J)] .= 0.0    
    Problem(num_layers, zeros(size(a)[1]), J, driver)
end


"""
    max_cut(graph::PyObject; num_layers::Int=1, driver=X)

Wrapper function setting up an instance of the MaxCut problem for the graph `graph`.

### Input
- `graph::PyObject`: The input graph, must be a [Python NetworkX](https://networkx.org/) graph.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The cost function for the MaxCut problem as defined in the [original QAOA paper](https://arxiv.org/abs/1411.4028) is

``
\\hat C = -\\frac{1}{2} \\sum_{(i, j) \\in E(G)} \\hat Z_i \\hat Z_j + \\mathrm{const.},
``
    
where ``E(G)`` is the set of edges of the graph ``G``.
"""
function max_cut(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)
    h = zeros(num_nodes)
    J = zeros(num_nodes, num_nodes)
    for edge in edges
        J[edge...] = -1/2.
    end
    
    Problem(num_layers, h, J, driver)
end


"""
    min_vertex_cover(graph::PyObject; num_layers::Int=1, driver=X)

Wrapper function setting up a problem instance for the minimum vertex cover of the graph `graph`.

### Input
- `graph::PyObject`: The input graph, must be a [Python NetworkX](https://networkx.org/) graph.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The cost function for the minimum-vertex-cover problem is 

``
\\hat C = -\\frac{3}{4} \\sum_{(i, j) \\in E(G)} (\\hat Z_i \\hat Z_j  +  \\hat Z_i  +  \\hat Z_j)  + \\sum_{i \\in V(G)} \\hat Z_i,
``

where ``E(G)`` is the set of edges and ``V(G)`` is the set of vertices of `graph` (we have a global minus sign since we _maximize_ the cost function).
"""
function min_vertex_cover(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)
    h = -ones(num_nodes)
    J = zeros(num_nodes, num_nodes)
    
    for edge in edges
        h[edge[1]] += 3/4.
        h[edge[2]] += 3/4.
        J[edge...] = 3/4.
    end    

    # note minus signs (we are maximizing)
    Problem(num_layers, -h, -J, driver)
end