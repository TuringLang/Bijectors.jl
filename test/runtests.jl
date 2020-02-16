using Bijectors, Random
using Test

Random.seed!(123456)

@testset "Interface" begin
    include("interface.jl")
end
include("transform.jl")
include("norm_flows.jl")
include("bijectors/permute.jl")
include("bijectors/couplings.jl")
