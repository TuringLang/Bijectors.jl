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

if GROUP == "All" || GROUP == "Interface"
    include("interface.jl")
    include("transform.jl")
    include("norm_flows.jl")
    include("bijectors/permute.jl")
    include("bijectors/leaky_relu.jl")
end

if !is_TRAVIS && (GROUP == "All" || GROUP == "AD")
    include("ad/utils.jl")
    include("ad/distributions.jl")
end

