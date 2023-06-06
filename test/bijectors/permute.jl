using Test

using Bijectors
using Bijectors: Permute

@testset "Permute" begin
    # Should fail because the permutation is non-injective
    # in the sense that the map is {1, 2} => {1}
    @test_throws ArgumentError Permute(2, 2 => 1)
    @test_throws ArgumentError Permute(2, [1, 2, 3] => [2, 1])

    # Simplest case
    b1 = Permute([
        0 1
        1 0
    ])
    b2 = Permute([2, 1])
    b3 = Permute(2, 2 => 1, 1 => 2)
    b4 = Permute(2, [1, 2] => [2, 1])

    @test b1.A == b2.A == b3.A == b4.A

    x = [1.0, 2.0]
    @test (inverse(b1) ∘ b1)(x) == x
    @test (inverse(b2) ∘ b2)(x) == x
    @test (inverse(b3) ∘ b3)(x) == x
    @test (inverse(b4) ∘ b4)(x) == x

    # Slightly more complex case; one entry is not permuted
    b1 = Permute([
        0 1 0
        1 0 0
        0 0 1
    ])
    b2 = Permute([2, 1, 3])
    b3 = Permute(3, 2 => 1, 1 => 2)
    b4 = Permute(3, [1, 2] => [2, 1])

    @test b1.A == b2.A == b3.A == b4.A

    x = [1.0, 2.0, 3.0]
    @test (inverse(b1) ∘ b1)(x) == x
    @test (inverse(b2) ∘ b2)(x) == x
    @test (inverse(b3) ∘ b3)(x) == x
    @test (inverse(b4) ∘ b4)(x) == x

    # logabsdetjac
    @test logabsdetjac(b1, x) == 0.0
    @test logabsdetjac(b2, x) == 0.0
    @test logabsdetjac(b3, x) == 0.0
    @test logabsdetjac(b4, x) == 0.0

    # with_logabsdet_jacobian
    y, logjac = with_logabsdet_jacobian(b1, x)
    @test (y == b1(x)) & (logjac == 0.0)

    y, logjac = with_logabsdet_jacobian(b2, x)
    @test (y == b2(x)) & (logjac == 0.0)

    y, logjac = with_logabsdet_jacobian(b3, x)
    @test (y == b3(x)) & (logjac == 0.0)

    y, logjac = with_logabsdet_jacobian(b4, x)
    @test (y == b4(x)) & (logjac == 0.0)
end
