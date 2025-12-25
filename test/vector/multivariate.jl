module VBMultivariateTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ..VectorTestUtils

dists = [
    Multinomial(10, [0.2, 0.5, 0.3]),
    MvNormal([0.0, 0.0], I),
    # TODO: MvNormalCanon
    # TODO: MvLogNormal
    # TODO: MvLogitNormal (returns a probability vector, so can use the same transform as Dirichlet)
    # TODO: Dirichlet
]

@testset "Multivariates" begin
    for d in dists
        VectorTestUtils.test_all(
            d; expected_zero_allocs=(to_vec, from_vec, to_linked_vec, from_linked_vec)
        )
    end
end

end # module VBUnivariateTests
