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
        our_method = sum(forward(flow, z)[2])

        @test our_method ≈ forward_diff
        @test inv(flow)(flow(z)) ≈ z
        @test (inv(flow) ∘ flow)(z) ≈ z
    end

    w = ones(10)
    u = zeros(10)
    b = 1.0
    flow = PlanarLayer(w, u, b)
    z = ones(10, 100)
    @test inv(flow)(flow(z)) ≈ z

    test_functor(flow, (w = w, u = u, b = b))
    test_functor(inv(flow), (orig = flow,))

    @testset "find_alpha" begin
        for wt_y in (-20.3, -3, -3//2, 0.0, 5, 29//4, 12.3)
            # the root finding algorithm assumes wt_u_hat ≥ -1 (satisfied for the flow)
            # |wt_u_hat| < eps checks that empty brackets are handled correctly
            # https://github.com/TuringLang/Bijectors.jl/issues/204
            for wt_u_hat in (-1, -1//2, -1e-20, 0, 1e-20, 3, 11//3, 17.2)
                for b in (-19.3, -8//3, -1, 0.0, 1//2, 3, 4.3)
                    # find α that solves wt_y = α + wt_u_hat * tanh(α + b)
                    α = @inferred(Bijectors.find_alpha(wt_y, wt_u_hat, b))

                    # check if α is an approximate solution to the considered equation
                    # have to set atol if wt_y is zero (otherwise only equality is checked)
                    @test wt_y ≈ α + wt_u_hat * tanh(α + b) atol=iszero(wt_y) ? 1e-14 : 0.0
                end
            end
        end

        # floating point issues
        # https://github.com/TuringLang/Bijectors.jl/issues/204
        wt_y = 0.8845640339582252
        wt_u_hat = 0.8296950433716855
        b = -1e8
        @test Bijectors.find_alpha(wt_y, wt_u_hat, b) ≈ wt_y + wt_u_hat
    end
end

@testset "RadialLayer" begin
    for i in 1:4
        flow = RadialLayer(2)
        z = randn(2, 20)
        forward_diff = log(abs(det(ForwardDiff.jacobian(t -> flow(t), z))))
        our_method = sum(forward(flow, z)[2])

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
    d = MvNormal(zeros(2), I)
    b = PlanarLayer(2)
    flow = transformed(d, b)  # <= Radial flow

    y = rand(flow)
    @test logpdf(flow, y) != 0.0

    x = rand(d)
    y = flow.transform(x)
    res = forward(flow.transform, x)
    lp = logpdf_forward(flow, x, res[2])

    @test res[1] ≈ y
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
