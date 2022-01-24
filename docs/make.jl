using Documenter
using Bijectors

makedocs(
    sitename = "Bijectors",
    format = Documenter.HTML(),
    modules = [Bijectors]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
