using Test
using Bijectors, ForwardDiff, LinearAlgebra
using Random: seed!

seed!(1)

@testset "InvertibleBatchNorm" begin
    z = randn(20, 2)
    flow = Bijectors.InvertibleBatchNorm(2)
    flow.active = false
    @test inv(inv(flow)) == flow 
    @test inv(flow)(flow(z)) ≈ z
    @test (inv(flow) ∘ flow)(z) ≈ z

    @test_throws AssertionError forward(flow, randn(2,10))
    
    @test logabsdetjac(inv(flow), flow(z)) ≈ - logabsdetjac(flow, z)

    y = flow(z)
    @test log(abs(det(ForwardDiff.jacobian(flow, z)))) ≈
     sum(logabsdetjac(flow, z)) rtol=1e-4 # fails return double
    @test log(abs(det(ForwardDiff.jacobian(inv(flow), y)))) ≈
     sum(logabsdetjac(inv(flow), y)) rtol=1e-4 # fails return double

    
end
