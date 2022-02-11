import Bijectors: OrderedBijector

@testset "OrderedBijector" begin
    b = OrderedBijector()

    # Length 1
    x = randn(1)
    test_bijector(b, x; test_not_identity=false)

    # Larger
    x = randn(5)
    test_bijector(b, x)

    y = b(x)
    @test sort(y) == y
end
