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
        "distributions.md",
        "types.md",
        "defining.md",
        "defining_examples.md",
        "advi.md",
        "flows.md",
        "vector.md",
    ],
    checkdocs=:exports,
    doctest=true,
    doctestfilters=[
        # Ignore the source of a warning in the doctest output, since this is dependent
        # on host. This is a line that starts with "└ @ " and ends with the line number.
        r"└ @ .+:[0-9]+",
    ],
)
