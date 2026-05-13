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
using Mooncake
using ReverseDiff

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

# Enzyme is tested separately in test/integration/enzyme.
const TEST_ADTYPES = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff(; compile=false)),
    ("ReverseDiffCompiled", AutoReverseDiff(; compile=true)),
    ("Mooncake", AutoMooncake()),
]

include("shared/ad_test_utils.jl")
include("shared/ad_bijector_tests.jl")
include("shared/vector_distributions.jl")

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
        include("bijectors/product_bijector.jl")
        include("bijectors/named_stacked.jl")
        include("distributionsad.jl")

        # These tests specifically check the implementation of AD backend rules.
        include("ad/chainrules.jl")
        include("ad/mooncake.jl")

        # These tests check that AD can differentiate through Bijectors # functionality without explicit rules.
        include("ad/flows.jl")
        include("ad/pd.jl")
        include("ad/corr.jl")
        include("ad/stacked.jl")
    end

    if GROUP == "All" || GROUP == "Vector"
        # VectorBijectors module.
        include("vector/univariate.jl")
        include("vector/multivariate.jl")
        include("vector/matrix.jl")
        include("vector/reshaped.jl")
        include("vector/cholesky.jl")
        include("vector/order.jl")
        include("vector/transformed.jl")
    end

    if GROUP == "All" || GROUP == "VectorProduct"
        # VectorBijectors module, part 2
        include("vector/product.jl")
    end

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
