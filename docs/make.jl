using Documenter
using Bijectors

# Doctest setup
DocMeta.setdocmeta!(Bijectors, :DocTestSetup, :(using Bijectors); recursive=true)

makedocs(
    sitename = "Bijectors",
    format = Documenter.HTML(),
    modules = [Bijectors]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/TuringLang/Bijectors.jl.git",
)
