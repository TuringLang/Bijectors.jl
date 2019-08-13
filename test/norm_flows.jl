using Test
using Bijectors, ForwardDiff, LinearAlgebra

@testset "planar flows" begin
    for i in 1:1
        flow = PlanarLayer(10)
        z = randn(10,100)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
        # @test inv(flow, transform(flow, z)) ≈ z
    end
end

@testset "radial flows" begin
    for i in 1:10
        flow = RadialLayer(10)
        z = randn(10,100)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
        # @test inv(flow, transform(flow, z)) ≈ z
    end
end
