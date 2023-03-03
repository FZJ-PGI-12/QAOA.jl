using Documenter, QAOA

makedocs(
    sitename="QAOA.jl",
    pages = [
                "Overview" => "index.md",
                "Examples" => ["examples/sherrington_kirkpatrick.md",
                               "examples/partition_problem.md",
                               "examples/max_cut.md",
                               "examples/min_vertex_cover.md"]

            ]
    )

# deploydocs(
#     repo = "github.com/NonequilibriumDynamics/KadanoffBaym.jl.git",
# )

