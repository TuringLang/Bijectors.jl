using Test
using Bijectors
using Bijectors: Exp, Log, Logit, AbstractNamedBijector, NamedBijector, NamedInverse, NamedCoupling

@testset "NamedBijector" begin
    b = NamedBijector((a = Exp(), b = Log()))
    @test b((a = 0.0, b = exp(1.0))) == (a = 1.0, b = 1.0)
end

@testset "NamedCoupling" begin
    nc = NamedCoupling(Val(:b), Val((:a, )), a -> Logit(zero(a), a))
    @inferred NamedCoupling(Val(:b), Val((:a, )), Shift)

    nc = NamedCoupling(:b, (:a, ), a -> Logit(0., a)) # <= not type-inferrable but eh

    @test Bijectors.target(nc) == :b
    @test Bijectors.deps(nc) == (:a, )

    @inferred Bijectors.target(nc)
    @inferred Bijectors.deps(nc)

    x = (a = 1.0, b = 0.5, c = 99999.)
    @test Bijectors.coupling(nc)(x.a) isa Logit
    @test inv(nc)(nc(x)) == x

    @test logabsdetjac(nc, x) == logabsdetjac(Logit(0., 1.), x.b)
    @test logabsdetjac(inv(nc), nc(x)) == -logabsdetjac(nc, x)

    x = (a = 0.0, b = 2.0, c = 1.0)
    nc = NamedCoupling(:c, (:a, :b), (a, b) -> Logit(a, b))
    @test nc(x).c == 0.0
    @test inv(nc)(nc(x)) == x
end
