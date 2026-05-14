@testset "Jacobians of SimplexBijector" begin
    b = SimplexBijector()
    ib = inverse(b)

    d_x = 10
    x = ib(randn(d_x - 1))
    y = b(x)

    @test Bijectors.jacobian(b, x) ≈ ForwardDiff.jacobian(b, x)
    @test Bijectors.jacobian(ib, y) ≈ ForwardDiff.jacobian(ib, y)

    # Just some additional computation so we also ensure the pullbacks are the same
    weights_x = randn(d_x)
    weights_y = randn(d_x - 1)

    # ForwardDiff.jl
    Δ_forwarddiff = ForwardDiff.gradient(z -> sum(weights_y .* b(z)), x)

    # ForwardDiff.jl
    Δ_forwarddiff_inv = ForwardDiff.gradient(z -> sum(weights_x .* ib(z)), y)
end
