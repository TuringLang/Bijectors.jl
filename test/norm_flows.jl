using Test
using Bijectors, ForwardDiff, LinearAlgebra

@testset "planar flows" begin
    for i in 1:10
        flow = PlanarLayer(2)
        z = randn(2, 1)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
    end

    w = ones(10, 1)
    u = zeros(10, 1)
    b = ones(1)
    flow = PlanarLayer(w, u, b)
    z = ones(10, 100)
    @test inv(flow, transform(flow, z)) ≈ z
end

@testset "radial flows" begin
    for i in 1:10
        flow = RadialLayer(1,0,zeros(2,1))
        z = randn(2, 1)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjacob)
        @test our_method ≈ forward_diff
    end
    α_ = ones(1)
    β = ones(1)
    z_0 = zeros(10, 1)
    z = ones(10, 100)
    flow = RadialLayer(α_, β, z_0)
    @test inv(flow, transform(flow, z)) ≈ z
end
