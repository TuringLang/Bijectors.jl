module VBProductTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

# These are purposely chosen because the vec_length output is the same but
# linked_vec_length differs.
m2 = MvNormal(zeros(2), I)
d2 = Dirichlet(ones(2))

products = [
    product_distribution(Normal()),
    product_distribution(Normal(), Normal()),
    product_distribution(Normal(), Beta(2, 2)),
    product_distribution(Beta(2, 2), Exponential()),
    product_distribution(m2, d2),
    product_distribution(m2, d2, m2, d2),
]

@testset "Product distributions" begin
    for d in products
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end
end

end # module VBReshapedTests
