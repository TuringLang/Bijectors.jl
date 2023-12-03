struct ProjectionBijector <: Bijectors.Bijector end

Bijectors.output_size(::ProjectionBijector, sz::Tuple{Int}) = (sz[1] - 1,)
Bijectors.output_size(::Inverse{ProjectionBijector}, sz::Int) = (sz[1] + 1,)

function Bijectors.with_logabsdet_jacobian(::ProjectionBijector, x::AbstractVector)
    return x[1:(end - 1)], 0
end
function Bijectors.with_logabsdet_jacobian(::Inverse{ProjectionBijector}, x::AbstractVector)
    return vcat(x, 0), 0
end

@testset "Stacked with differing input and output size" begin
    bs = [
        Stacked((elementwise(exp), ProjectionBijector()), (1:1, 2:3)),
        Stacked([elementwise(exp), ProjectionBijector()], [1:1, 2:3]),
        Stacked([elementwise(exp), ProjectionBijector()], (1:1, 2:3)),
        Stacked((elementwise(exp), ProjectionBijector()), [1:1, 2:3]),
    ]
    @testset "$b" for b in bs
        binv = inverse(b)
        x = [1.0, 2.0, 3.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (3,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (3,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0]
    end

    @testset "composition" begin
        # Composition with one dimension reduction.
        b = Stacked((elementwise(exp), ProjectionBijector() ∘ identity), [1:1, 2:3])
        binv = inverse(b)
        x = [1.0, 2.0, 3.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (3,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (3,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0]

        # Composition with two dimension reductions.
        b = Stacked((elementwise(exp), ProjectionBijector() ∘ ProjectionBijector()), [1:1, 2:4])
        binv = inverse(b)
        x = [1.0, 2.0, 3.0, 4.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (4,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (4,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0, 0.0]
    end
end
