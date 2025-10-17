module BijectorsNamedStackedTests

using Bijectors
using Test
using LinearAlgebra: Cholesky, I

isapprox_nested(a::Number, b::Number; kwargs...) = isapprox(a, b; kwargs...)
isapprox_nested(a::AbstractVector, b::AbstractVector; kwargs...) = isapprox(a, b; kwargs...)
function isapprox_nested(a::Cholesky, b::Cholesky; kwargs...)
    return isapprox(a.U, b.U; kwargs...) && isapprox(a.L, b.L; kwargs...)
end
function isapprox_nested(a::NamedTuple, b::NamedTuple; kwargs...)
    keys(a) == keys(b) || return false
    return all(k -> isapprox_nested(a[k], b[k]; kwargs...), keys(a))
end

@testset "NamedStacked" begin
    # The first boolean indicates whether bijector(dist) is type stable.
    test_dists = [
        (true, product_distribution((a=Normal(), b=LogNormal()))),
        (
            true,
            product_distribution((
                a=Beta(2, 2),
                b=LKJCholesky(3, 1.0),
                c=InverseGamma(2, 3),
                d=MvNormal(zeros(3), I),
            )),
        ),
        (true, product_distribution((; a=LogNormal()))),
        # bijector(d) cannot be type stable because bijector(d.b) is itself not type stable.
        # This should probably be fixed but it's not NamedStacked's fault.
        (
            false,
            product_distribution((
                a=LogNormal(), b=product_distribution([LogNormal(), InverseGamma(2, 3)])
            )),
        ),
        (
            true,
            product_distribution((
                a=LogNormal(), b=product_distribution((x=LogNormal(), y=InverseGamma(2, 3)))
            )),
        ),
    ]

    @testset "transforms product distribution $i" for (i, (is_typestable, d)) in
                                                      enumerate(test_dists)
        b = bijector(d)
        if is_typestable
            @inferred bijector(d)
        end

        x = rand(d)
        y = transform(b, x)
        @inferred transform(b, x)
        @inferred b(x)

        binv = inverse(b)
        @inferred inverse(b)
        x2 = transform(binv, y)
        @inferred transform(binv, y)
        @inferred binv(y)

        @test isapprox_nested(x, x2)
    end

    @testset "jacobians" begin
        @testset "non-nested" begin
            dist = product_distribution((a=Normal(), b=LogNormal()))
            b = bijector(dist)
            x = rand(dist)
            y, logjac = with_logabsdet_jacobian(b, x)
            @inferred with_logabsdet_jacobian(b, x)

            # since bijector(Normal()) is identity, the only Jacobian term should
            # come from LogNormal
            expected_logjac = logabsdetjac(b.transforms.b, x.b)
            @test isapprox(logjac, expected_logjac)

            x2, inv_logjac = with_logabsdet_jacobian(inverse(b), y)
            @inferred with_logabsdet_jacobian(inverse(b), y)
            @test isapprox_nested(x, x2)
            @test isapprox(inv_logjac, -expected_logjac)

            # check logabsdetjac as well
            @test isapprox(logabsdetjac(b, x), expected_logjac)
            @test isapprox(logabsdetjac(inverse(b), y), -expected_logjac)
        end

        @testset "nested" begin
            dist = product_distribution((
                a=LogNormal(), b=product_distribution((x=LogNormal(), y=InverseGamma(2, 3)))
            ))
            b = bijector(dist)
            x = rand(dist)
            y, logjac = with_logabsdet_jacobian(b, x)
            @inferred with_logabsdet_jacobian(b, x)

            expected_logjac =
                logabsdetjac(b.transforms.a, x.a) +
                logabsdetjac(b.transforms.b.transforms.x, x.b.x) +
                logabsdetjac(b.transforms.b.transforms.y, x.b.y)
            @test isapprox(logjac, expected_logjac)

            x2, inv_logjac = with_logabsdet_jacobian(inverse(b), y)
            @inferred with_logabsdet_jacobian(inverse(b), y)
            @test isapprox_nested(x, x2)
            @test isapprox(inv_logjac, -expected_logjac)

            # check logabsdetjac as well
            @test isapprox(logabsdetjac(b, x), expected_logjac)
            @test isapprox(logabsdetjac(inverse(b), y), -expected_logjac)
        end
    end

    @testset "output size on ProductNamedTupleDistribution" begin
        d = product_distribution((a=Normal(), b=LogNormal()))
        b = bijector(d)
        @test output_size(b, d) == (2,)

        d = product_distribution((
            a=LogNormal(), b=product_distribution((x=LogNormal(), y=InverseGamma(2, 3)))
        ))
        b = bijector(d)
        @test output_size(b, d) == (3,)
    end
end

end
