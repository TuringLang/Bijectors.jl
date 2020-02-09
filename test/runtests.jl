using Bijectors, Random
using Test

Random.seed!(123456)

@testset "Interface" begin
    include("interface.jl")
end
include("transform.jl")
@testset "Leaky ReLU" begin
    include("bijectors/leaky_relu.jl")
end
