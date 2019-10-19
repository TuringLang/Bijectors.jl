using Bijectors, Random

Random.seed!(123456)

include("interface.jl")
include("transform.jl")
include("norm_flows.jl")
include("bijectors/permute.jl")
