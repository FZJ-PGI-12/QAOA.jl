using Documenter, QAOA, Graphs

makedocs(
    sitename = "QAOA.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    warnonly = true,
    checkdocs = :exports,
    pages = [
        "Overview" => "index.md",
        "Examples" => ["examples/max_cut.md",
                       "examples/sherrington_kirkpatrick.md",
                       "examples/partition_problem.md",
                       "examples/min_vertex_cover.md",
                       "examples/constraints.md",
                       "examples/schedules.md"],
        "Performance" => "performance.md",
        "Citation" => "citation.md",
        "Support and Contributing" => "support.md",
    ],
)

deploydocs(
    repo = "github.com/FZJ-PGI-12/QAOA.jl.git",
    devbranch = "master",
)
