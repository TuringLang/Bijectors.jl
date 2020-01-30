using DistributionsAD, Bijectors, Random
using Test

Random.seed!(123456)

@testset "Interface" begin
    include("interface.jl")
end
include("transform.jl")
