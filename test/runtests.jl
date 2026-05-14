using Bijectors

using ChainRulesTestUtils
using Combinatorics
using DifferentiationInterface
using DistributionsAD
using Documenter: Documenter
using FiniteDifferences
using ForwardDiff
using Functors
using LogExpFunctions

using Random, LinearAlgebra, Test

using Bijectors:
    Shift,
    Scale,
    Logit,
    SimplexBijector,
    PDBijector,
    Permute,
    PlanarLayer,
    RadialLayer,
    Stacked,
    TruncatedBijector

using ChangesOfVariables: ChangesOfVariables
using InverseFunctions: InverseFunctions
using LazyArrays: LazyArrays

const GROUP = get(ENV, "GROUP", "All")

# AD backends are tested separately in test/integration/{enzyme,mooncake,reversediff}.

include("test_resources.jl")

# Always include this since it can be useful for other tests.
include("bijectors/utils.jl")

@testset "Bijectors.jl" begin
    if GROUP == "All" || GROUP == "Classic"
        include("interface.jl")
        include("transform.jl")
        include("norm_flows.jl")
        include("bijectors/permute.jl")
        include("bijectors/rational_quadratic_spline.jl")
        include("bijectors/named_bijector.jl")
        include("bijectors/leaky_relu.jl")
        include("bijectors/coupling.jl")
        include("bijectors/ordered.jl")
        include("bijectors/pd.jl")
        include("bijectors/reshape.jl")
        include("bijectors/corr.jl")
        include("bijectors/chainrules.jl")
        include("bijectors/product_bijector.jl")
        include("bijectors/named_stacked.jl")
        include("distributionsad.jl")
    end

    # Vector test_all coverage lives inside `vector_bijectors.jl`, which is included
    # transitively via `test_resources.jl` and runs its `@testset` if `GROUP` is set.

    if GROUP == "All" || GROUP == "Doctests"
        @testset "doctests" begin
            Documenter.DocMeta.setdocmeta!(
                Bijectors, :DocTestSetup, :(using Bijectors); recursive=true
            )
            doctestfilters = [
                # Ignore the source of a warning in the doctest output, since this is dependent
                # on host. This is a line that starts with "└ @ " and ends with the line number.
                r"└ @ .+:[0-9]+",
            ]
            Documenter.doctest(Bijectors; manual=false, doctestfilters=doctestfilters)
        end
    end
end
