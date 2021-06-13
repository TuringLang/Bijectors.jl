import Bijectors: OrderedBijector

@testset "OrderedBijector" begin
    b = OrderedBijector()

    # Length 1
    x = randn(1)
    y = b(x)
    test_bijector(b, hcat(x, x), hcat(y, y), zeros(2))

    # Larger
    x = randn(5)
    test_bijector(b, hcat(x, x))

    y = b(x)
    @test sort(y) == y

    ys = b(hcat(x, x))
    @test sort(ys[:, 1]) == ys[:, 1]
    @test sort(ys[:, 2]) == ys[:, 2]
end
