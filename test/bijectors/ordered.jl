import Bijectors: OrderedBijector, ordered
using LinearAlgebra

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

@testset "ordered" begin
    @testset "$d" for d in [
        MvNormal(1:5, Diagonal(6:10)),
        MvTDist(1, collect(1:5), Matrix(I(5))),
        product_distribution(fill(Normal(), 5)),
        product_distribution(fill(TDist(1), 5))
    ]
        d_ordered = ordered(d)
        @test d_ordered isa Bijectors.TransformedDistribution
        @test d_ordered.dist === d
        @test d_ordered.transform isa OrderedBijector
        y = randn(5)
        x = inverse(bijector(d_ordered))(y)
        @test issorted(x)
    end

    @testset "non-identity bijector is not supported" begin
        d = Dirichlet(ones(5))
        @test_throws ArgumentError ordered(d)
    end
end
