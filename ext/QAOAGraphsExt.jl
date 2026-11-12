"""
    QAOAGraphsExt

Graph problems built from a Graphs.jl graph, loaded with `using Graphs`.
"""
module QAOAGraphsExt

using QAOA
using Graphs: AbstractGraph, edges, nv, src, dst, weights

"""
    _edge_list(g) -> Vector{Tuple{Int,Int}}

The edges of `g` as node pairs.
"""
_edge_list(g::AbstractGraph) = [(src(e), dst(e)) for e in edges(g)]

"""
    max_cut(g::AbstractGraph) -> IsingProblem

The MaxCut problem on the graph `g`, with the edge weights of `g` (1 for an
unweighted graph); see [`max_cut`](@ref).
"""
function QAOA.max_cut(g::AbstractGraph)
    pairs = _edge_list(g)
    w = weights(g)
    QAOA.max_cut(nv(g), pairs; weights = [w[i, j] for (i, j) in pairs])
end

"""
    min_vertex_cover(g::AbstractGraph; penalty = 2) -> QUBOProblem

The minimum vertex cover of the graph `g`; see [`min_vertex_cover`](@ref).
"""
QAOA.min_vertex_cover(g::AbstractGraph; penalty::Real = 2) =
    QAOA.min_vertex_cover(nv(g), _edge_list(g); penalty)

end
