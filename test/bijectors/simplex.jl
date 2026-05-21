@testset "Jacobians of SimplexBijector" begin
    b = SimplexBijector()
    ib = inverse(b)

    d_x = 10
    x = ib(randn(d_x - 1))
    y = b(x)

    @test Bijectors.jacobian(b, x) ≈ ForwardDiff.jacobian(b, x)
    @test Bijectors.jacobian(ib, y) ≈ ForwardDiff.jacobian(ib, y)
end
