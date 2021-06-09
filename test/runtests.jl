using Bijectors

using ChainRulesTestUtils
using Combinatorics
using DistributionsAD
using FiniteDifferences
using ForwardDiff
using Functors
using ReverseDiff
using Tracker
using Zygote

using Random, LinearAlgebra, Test

using Bijectors: Log, Exp, Shift, Scale, Logit, SimplexBijector, PDBijector, Permute,
    PlanarLayer, RadialLayer, Stacked, TruncatedBijector, ADBijector

using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal,
    TuringPoissonBinomial

import NNlib

const GROUP = get(ENV, "GROUP", "All")

# Always include this since it can be useful for other tests.
include("ad/utils.jl")
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
end

if GROUP == "All" || GROUP == "AD"
    include("ad/chainrules.jl")
    include("ad/flows.jl")
    include("ad/distributions.jl")
    if AD == "All" || AD == "ReverseDiff" 
        @testset "Turing issue 1385" begin
            dist = arraydist(truncated.(Laplace.(0, [1, 2]), -10.0, 70.0))
            x = ReverseDiff.track(rand(dist))
            @test typeof(bijector(dist)(x)) <: ReverseDiff.TrackedArray
        end
    end
end
