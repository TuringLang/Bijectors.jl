using Documenter
using Bijectors

# Doctest setup
DocMeta.setdocmeta!(Bijectors, :DocTestSetup, :(using Bijectors); recursive=true)

makedocs(;
    sitename="Bijectors",
    format=Documenter.HTML(),
    modules=[Bijectors],
    pages=[
        "index.md",
        "interface.md",
        "defining.md",
        "distributions.md",
        "types.md",
        "advi.md",
        "flows.md",
    ],
    checkdocs=:exports,
    doctest=false,
)
