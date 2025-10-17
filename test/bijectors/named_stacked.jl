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

    @testset "product distribution $i" for (i, (is_typestable, d)) in enumerate(test_dists)
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
end

end
