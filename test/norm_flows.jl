using Test
using Bijectors, ForwardDiff, LinearAlgebra

@testset "planar flows" begin
    for i in 1:100
        flow = PlanarLayer(10)
        z = randn(10,100)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t).data, z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
    end
end

@testset "radial flows" begin
    for i in 1:100
        flow = RadialLayer(10)
        z = randn(10,100)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t).data, z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
    end
end
