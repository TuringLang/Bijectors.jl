using ADTypes
using Bijectors

using ChainRulesTestUtils
using Combinatorics
using AbstractPPL
using DistributionsAD
using Documenter: Documenter
using FiniteDifferences
using ForwardDiff
using Functors
using LogExpFunctions
using Mooncake
using ReverseDiff
using Pkg

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
# Enzyme doesn't work on 1.12 yet
const TEST_ENZYME = VERSION < v"1.12.0"

TEST_ADTYPES = [("ForwardDiff", AutoForwardDiff()), ("Mooncake", AutoMooncake())]
if TEST_ENZYME
    Pkg.add("Enzyme")
    Pkg.add("EnzymeTestUtils")
    using Enzyme: Enzyme, set_runtime_activity, Forward, Reverse, Const
    TEST_ADTYPES = [
        TEST_ADTYPES...,
        (
            "EnzymeForward",
            AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
        ),
        (
            "EnzymeReverse",
            AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
        ),
    ]
end
const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))

function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
    @info "testing AD for function $f with $backend"
    ref_prepared = AbstractPPL.prepare(REF_BACKEND, f, x)
    prepared = AbstractPPL.prepare(backend, f, x)
    _, ref_gradient = AbstractPPL.value_and_gradient(ref_prepared, x)
    _, gradient = AbstractPPL.value_and_gradient(prepared, x)
    @test isapprox(gradient, ref_gradient; rtol=rtol, atol=atol)
end

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
        if TEST_ENZYME
            include("ad/enzyme.jl")
        end

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
