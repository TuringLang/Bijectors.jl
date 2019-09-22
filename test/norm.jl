using Test
using Bijectors, ForwardDiff, LinearAlgebra
using Random: seed!

seed!(1)

@testset "BatchNormFlow" begin
    z = randn(2, 20)
    flow = Bijectors.BatchNormFlow(2)
    flow.active=false
    forward_diff = log(abs(det(ForwardDiff.jacobian(t -> flow(t), z))))
    our_method = sum(forward(flow, z).logabsdetjacob)
    @test inv(flow)(flow(z)) ≈ z
    @test (inv(flow) ∘ flow)(z) ≈ z
    @test our_method ≈ forward_diff rtol=1e-4
end
