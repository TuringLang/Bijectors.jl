module BijectorsNamedStackedTests

using Bijectors
using Distributions
using Test

@testset "NamedStacked" begin
    d = product_distribution((a=Normal(), b=LogNormal()))
    b = bijector(d)
    @inferred bijector(d)

    x = rand(d)
    y = transform(b, x)
    @inferred transform(b, x)
    @inferred b(x)

    binv = inverse(b)
    @inferred inverse(b)
    x2 = transform(binv, y)
    @inferred transform(binv, y)
    @inferred binv(y)

    @test isapprox(x.a, x2.a)
    @test isapprox(x.b, x2.b)
end

end
