import Bijectors: OrderedBijector

@testset "OrderedBijector" begin
    b = OrderedBijector()

    # Length 1
    x = randn(1)
    y = b(x)
    test_bijector(b, hcat(x, x), hcat(y, y), zeros(2))

    # Larger
    x = randn(5)
    xs = hcat(x, x)
    test_bijector(b, xs)

    y = b(x)
    @test sort(y) == y

    ys = b(xs)
    @test sort(ys[:, 1]) == ys[:, 1]
    @test sort(ys[:, 2]) == ys[:, 2]

    # `ChainRules`
    test_rrule(Bijectors._transform_ordered, x)
    test_rrule(Bijectors._transform_ordered, xs)
    test_rrule(Bijectors._transform_inverse_ordered, y)
    test_rrule(Bijectors._transform_inverse_ordered, ys)
end
