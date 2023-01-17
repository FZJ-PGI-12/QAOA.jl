using PyCall
np = pyimport("numpy")
nx = pyimport("networkx")

function sherrington_kirkpatrick()
    0
end

function partition_problem()
    0
end

function max_cut()
    0
end


"""
    min_vertex_cover(graph::PyObject; num_layers::Int=1, driver=X)

Wrapper function setting up a problem instance for the minimum vertex cover of the graph `graph`.

### Input
- `graph::PyObject`: The input graph, must be a Python NetworkX graph.
- `num_layers::Int=1` (optional): The number of QAOA layers usually denoted by ``p``.
- `driver=X`` (optional): The driver or mixer used in the QAOA.

### Output
- An instance of the `Problem` struct holding all relevant quantities.

### Notes
The cost function for the minimum-vertex-cover problem is 

``
\hat C = -\frac{3}{4} \sum_{(i, j) \in E(G)} (\hat Z_i \hat Z_j  +  \hat Z_i  +  \hat Z_j)  + \sum_{i \in V(G)} \hat Z_i,
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