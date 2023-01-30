import Bijectors: OrderedBijector, ordered
using LinearAlgebra

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
end

@testset "ordered" begin
    d = MvNormal(1:5, Diagonal(6:10))
    d_ordered = ordered(d)
    @test d_ordered isa Bijectors.TransformedDistribution
    @test d_ordered.dist === d
    @test d_ordered.transform isa OrderedBijector
    y = randn(5)
    x = inv(bijector(d_ordered))(y)
    @test issorted(x)

    d = Product(fill(Normal(), 5))
    # currently errors because `bijector(Product(fill(Normal(), 5)))` is not an `Identity`
    @test_broken ordered(d) isa Bijectors.TransformedDistribution

    # non-Identity bijector is not supported
    d = Dirichlet(ones(5))
    @test_throws ArgumentError ordered(d)
end
