module VBCholeskyTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

dists = [
    # Note: can't test LKJCholesky(1, ...) because its linked vector is length-zero and
    # DifferentiationInterface trips up with empty vectors.
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

@testset "Cholesky" begin
    for d in dists
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end
end

end # module VBMultivariateTests
