using Documenter, QAOA

makedocs(
    sitename="QAOA.jl",
    pages = [
                "Overview" => "index.md",
                "Examples" => ["examples/sherrington_kirkpatrick.md",
                               "examples/partition_problem.md",
                               "examples/max_cut.md",
                               "examples/min_vertex_cover.md",
                               "examples/mean_field.md"],
                "Citation" => "citation.md",
                "Support and Contributing" => "support.md"

            ]
    )

deploydocs(
    repo = "github.com/FZJ-PGI-12/QAOA.jl.git",
)

