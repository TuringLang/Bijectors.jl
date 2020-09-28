using Bijectors

using Combinatorics
using DistributionsAD
using FiniteDiff
using ForwardDiff
using ReverseDiff
using Tracker
using Zygote

using Random, LinearAlgebra, Test

using Bijectors: Log, Exp, Shift, Scale, Logit, SimplexBijector, PDBijector, Permute,
    PlanarLayer, RadialLayer, Stacked, TruncatedBijector, ADBijector

using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal,
    TuringPoissonBinomial

const is_TRAVIS = haskey(ENV, "TRAVIS")
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

if !is_TRAVIS && (GROUP == "All" || GROUP == "AD")
    include("ad/distributions.jl")
end

