using Bijectors

using ChainRulesTestUtils
using Combinatorics
using DistributionsAD
using FiniteDifferences
using ForwardDiff
using Functors
using LogExpFunctions
using ReverseDiff
using Tracker
using Zygote

using Random, LinearAlgebra, Test

using Bijectors: Log, Exp, Shift, Scale, Logit, SimplexBijector, PDBijector, Permute,
    PlanarLayer, RadialLayer, Stacked, TruncatedBijector, ADBijector

using ChangesOfVariables: test_with_logabsdet_jacobian
using InverseFunctions: test_inverse

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
    include("bijectors/ordered.jl")
end

if GROUP == "All" || GROUP == "AD"
    include("ad/chainrules.jl")
    include("ad/flows.jl")
end

