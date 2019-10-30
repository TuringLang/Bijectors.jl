using Bijectors, Random

Random.seed!(123456)

include("interface.jl")
include("transform.jl")
include("batch_norm.jl")
