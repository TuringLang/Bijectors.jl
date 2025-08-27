using Documenter
using Bijectors

# Doctest setup
DocMeta.setdocmeta!(Bijectors, :DocTestSetup, :(using Bijectors); recursive=true)

makedocs(;
    sitename="Bijectors",
    format=Documenter.HTML(
        edit_link="main"
    ),
    modules=[Bijectors],
    pages=[
        "Home" => "index.md",
        "Transforms" => "transforms.md",
        "Distributions.jl integration" => "distributions.md",
        "Examples" => "examples.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/TuringLang/Bijectors.jl.git",
    devbranch="main",
)
