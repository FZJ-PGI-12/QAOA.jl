using PyCall
np = pyimport("numpy")
nx = pyimport("networkx")

function sherrington_kirkpatrick()
    0
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
    J = -2 * np.outer(a |> transpose, a)
    np.fill_diagonal(J, 0.)  
    Problem(num_layers, zeros(size(a)[1]), J, driver)
end


"""
    max_cut(graph::PyObject; num_layers::Int=1, driver=X)

Wrapper function setting up an instance of the MaxCut problem for the graph `graph`.

### Input
- `graph::PyObject`: The input graph, must be a Python NetworkX graph.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The cost function for the MaxCut problem as defined in the [original QAOA paper](https://arxiv.org/abs/1411.4028) is

``
\\hat C = \\frac{1}{2} \\sum_{(i, j) \\in E(G)} (1 - \\hat Z_i \\hat Z_j),
``
    
where ``E(G)`` is the set of edges of the graph ``G``.
"""
function max_cut(graph::PyObject; num_layers::Int=1, driver=X)
    @assert pybuiltin(:isinstance)(graph, (nx.Graph)) "Input must be a Python NetworkX graph."

    N = graph.number_of_nodes()
    h = zeros(N)
    J = zeros(N, N)
    for edge in graph.edges
        J[(edge .+ (1, 1))...] = -1/2.
    end
    
    Problem(num_layers, h, J, driver)
end


"""
    min_vertex_cover(graph::PyObject; num_layers::Int=1, driver=X)

Wrapper function setting up a problem instance for the minimum vertex cover of the graph `graph`.

### Input
- `graph::PyObject`: The input graph, must be a Python NetworkX graph.
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
function min_vertex_cover(graph::PyObject; num_layers::Int=1, driver=X)
    @assert pybuiltin(:isinstance)(graph, (nx.Graph)) "Input must be a Python NetworkX graph."

    N = graph.number_of_nodes()
    h = -ones(N)
    J = zeros(N, N)
    
    for edge in graph.edges
        h[edge[1] + 1] += 3/4.
        h[edge[2] + 1] += 3/4.
        J[(edge .+ (1, 1))...] = 3/4.
    end    

    # note minus signs (necessary when maximizing)
    Problem(num_layers, -h, -J, driver)
end