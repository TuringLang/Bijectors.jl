using Test
using Bijectors, ForwardDiff, LinearAlgebra
using Random: seed!

seed!(1)

@testset "PlanarLayer" begin
    for i in 1:4
        flow = PlanarLayer(2)
        z = randn(2, 20)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjac)
        @test our_method ≈ forward_diff
        @test inv(flow, transform(flow, z)) ≈ z rtol=0.2
    end

    w = ones(10, 1)
    u = zeros(10, 1)
    b = ones(1)
    flow = PlanarLayer(w, u, b)
    z = ones(10, 100)
    @test inv(flow, transform(flow, z)) ≈ z
end

@testset "RadialLayer" begin
    for i in 1:4
        flow = RadialLayer(2)
        z = randn(2, 20)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> transform(flow, t), z))))
        our_method = sum(forward(flow, z).logabsdetjac)
        @test our_method ≈ forward_diff
        @test inv(flow, transform(flow, z)) ≈ z
    end

    α_ = ones(1)
    β = ones(1)
    z_0 = zeros(10, 1)
    z = ones(10, 100)
    flow = RadialLayer(α_, β, z_0)
    @test inv(flow, transform(flow, z)) ≈ z
end
