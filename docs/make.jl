using Documenter, QAOA

makedocs(
    sitename="QAOA.jl",
    pages = [
                "Overview" => "index.md",
                "Examples" => ["examples/mean_field.md",
                               "examples/prime_number.md",
                               "QAOA" => ["examples/QAOA/sherrington_kirkpatrick.md",
                               "examples/QAOA/partition_problem.md",
                               "examples/QAOA/max_cut.md",
                               "examples/QAOA/min_vertex_cover.md"],
                               "examples/annealing.md"],
                "Citation" => "citation.md",
                "Support and Contributing" => "support.md"

            ]
    )

deploydocs(
    repo = "github.com/FZJ-PGI-12/QAOA.jl.git",
)

