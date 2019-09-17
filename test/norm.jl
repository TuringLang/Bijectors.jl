using Test
using Bijectors, ForwardDiff, LinearAlgebra
using Random: seed!

seed!(1)

@testset "BatchNorm" begin
    z = randn(2, 20)
    flow = Bijectors.BatchNorm(2)
    forward_diff = log(abs(det(ForwardDiff.jacobian(t -> flow(t), z))))
    our_method = sum(forward(flow, z).logabsdetjacob)

    flow.active=false
    @test inv(flow)(flow(z)) ≈ z rtol=0.2
    @test (inv(flow) ∘ flow)(z) ≈ z rtol=0.2
    @test our_method ≈ forward_diff
end
