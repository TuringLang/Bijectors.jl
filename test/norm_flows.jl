using Test
using Bijectors, ForwardDiff, LinearAlgebra
using Random: seed!

seed!(1)

@testset "InvertibleBatchNorm" begin
    x = randn(2, 20)
    bn = InvertibleBatchNorm(2)

    @test inv(inv(bn)) == bn
    @test inv(bn)(bn(x)) ≈ x
    @test (inv(bn) ∘ bn)(x) ≈ x
    @test_throws ErrorException forward(bn, randn(10,2))
    @test logabsdetjac(inv(bn), bn(x)) ≈ - logabsdetjac(bn, x)

    y, ladj = forward(bn, x)
    @test log(abs(det(ForwardDiff.jacobian(bn, x)))) ≈ sum(ladj)
    @test log(abs(det(ForwardDiff.jacobian(inv(bn), y)))) ≈ sum(logabsdetjac(inv(bn), y))

    test_functor(bn, (b = bn.b, logs = bn.logs))
end

@testset "PlanarLayer" begin
    for i in 1:4
        flow = PlanarLayer(2)
        z = randn(2, 20)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> flow(t), z))))
        our_method = sum(forward(flow, z).logabsdetjac)

        @test our_method ≈ forward_diff
        @test inv(flow)(flow(z)) ≈ z rtol=0.25
        @test (inv(flow) ∘ flow)(z) ≈ z rtol=0.25
    end

    w = ones(10)
    u = zeros(10)
    b = 1.0
    flow = PlanarLayer(w, u, b)
    z = ones(10, 100)
    @test inv(flow)(flow(z)) ≈ z

    test_functor(flow, (w = w, u = u, b = b))
    test_functor(inv(flow), (orig = flow,))
end

@testset "RadialLayer" begin
    for i in 1:4
        flow = RadialLayer(2)
        z = randn(2, 20)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> flow(t), z))))
        our_method = sum(forward(flow, z).logabsdetjac)

        @test our_method ≈ forward_diff
        @test inv(flow)(flow(z)) ≈ z rtol=0.2
        @test (inv(flow) ∘ flow)(z) ≈ z rtol=0.2
    end

    α_ = 1.0
    β = 1.0
    z_0 = zeros(10)
    z = ones(10, 100)
    flow = RadialLayer(α_, β, z_0)
    @test inv(flow)(flow(z)) ≈ z

    test_functor(flow, (α_ = α_, β = β, z_0 = z_0))
    test_functor(inv(flow), (orig = flow,))
end

@testset "Flows" begin
    d = MvNormal(zeros(2), ones(2))
    b = PlanarLayer(2)
    flow = transformed(d, b)  # <= Radial flow

    y = rand(flow)
    @test logpdf(flow, y) != 0.0

    x = rand(d)
    y = flow.transform(x)
    res = forward(flow.transform, x)
    lp = logpdf_forward(flow, x, res.logabsdetjac)

    @test res.rv ≈ y
    @test logpdf(flow, y) ≈ lp rtol=0.1

    # flow with unconstrained-to-constrained
    d1 = Beta()
    b1 = inv(bijector(d1))
    d2 = InverseGamma()
    b2 = inv(bijector(d2))

    x = rand(d) .+ 10
    y = b(x)

    sb = stack(b1, b1)
    @test all((sb ∘ b)(x) .≤ 1.0)

    sb = stack(b1, b2)
    cb = (sb ∘ b)
    y = cb(x)
    @test (0 ≤ y[1] ≤ 1.0) && (0 < y[2])
end
