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
# Mooncake and Enzyme don't work on 1.12 yet
const TEST_ENZYME_AND_MOONCAKE = VERSION < v"1.12.0"

TEST_ADTYPES = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff(; compile=false)),
    ("ReverseDiffCompiled", AutoReverseDiff(; compile=true)),
]
if TEST_ENZYME_AND_MOONCAKE
    Pkg.add("Enzyme")
    Pkg.add("Mooncake")
    using Enzyme: Enzyme, set_runtime_activity, Forward, Reverse, Const
    using Mooncake
    TEST_ADTYPES = [
        TEST_ADTYPES...,
        ("Mooncake", AutoMooncake()),
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

# Always include this since it can be useful for other tests.
include("bijectors/utils.jl")

if GROUP == "All" || GROUP == "Interface"
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
    include("distributionsad.jl")
end

if GROUP == "All" || GROUP == "AD"
    # These tests specifically check the implementation of AD backend rules.
    include("ad/chainrules.jl")
    if TEST_ENZYME_AND_MOONCAKE
        include("ad/enzyme.jl")
        include("ad/mooncake.jl")
    end

    # These tests check that AD can differentiate through Bijectors
    # functionality without explicit rules.
    const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))
    function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
        @info "testing AD for function $f with $backend"
        ref_gradient = DifferentiationInterface.gradient(f, REF_BACKEND, x)
        gradient = DifferentiationInterface.gradient(f, backend, x)
        @test isapprox(gradient, ref_gradient; rtol=rtol, atol=atol)
    end
    include("ad/flows.jl")
    include("ad/pd.jl")
    include("ad/corr.jl")
    include("ad/stacked.jl")
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
