module VBMultivariateTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

multivariates = [
    # identity transforms (discrete multivariate)
    Multinomial(10, [0.2, 0.5, 0.3]),
    # identity transforms (continuous multivariate)
    MvNormal([0.0, 0.0], I),
    MvNormalCanon([1.0, 2.0, 3.0], [4.0 -2.0 -1.0; -2.0 5.0 -1.0; -1.0 -1.0 6.0]),
    # broadcast exp/log
    MvLogNormal([0.0, 0.0], I),
    # simplex distribution
    MvLogitNormal([1.0, 2.0], Diagonal([4.0, 5.0])),
    Dirichlet([2.0, 3.0, 5.0]),
]

@testset "Multivariates" begin
    for d in multivariates
        expected_zero_allocs = if d isa Union{Dirichlet,MvLogitNormal,MvLogNormal}
            (to_vec, from_vec)
        else
            (to_vec, from_vec, to_linked_vec, from_linked_vec)
        end
        VectorBijectors.test_all(d; expected_zero_allocs=expected_zero_allocs)
    end
end

end # module VBMultivariateTests
