using Bijectors

using ChainRulesTestUtils
using Combinatorics
using DifferentiationInterface
using DistributionsAD
using Documenter: Documenter
using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using ForwardDiff
using Functors
using LogExpFunctions
using Mooncake
using ReverseDiff
using Tracker

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
const IS_PRERELEASE = !isempty(VERSION.prerelease)
TEST_ADTYPES = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff(; compile=false)),
    ("ReverseDiffCompiled", AutoReverseDiff(; compile=true)),
]
if !IS_PRERELEASE
    push!(TEST_ADTYPES, ("Mooncake", AutoMooncake()))
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
    const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))
    function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
        ref_gradient = DifferentiationInterface.gradient(f, REF_BACKEND, x)
        gradient = DifferentiationInterface.gradient(f, backend, x)
        @test isapprox(gradient, ref_gradient; rtol=rtol, atol=atol)
    end
    include("ad/chainrules.jl")
    include("ad/flows.jl")
    include("ad/pd.jl")
    include("ad/corr.jl")
    include("ad/stacked.jl")
    include("ad/enzyme.jl")
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
