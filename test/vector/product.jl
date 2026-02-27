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
    # Tuples
    product_distribution(Normal()),
    product_distribution(Normal(), Normal()),
    product_distribution(Normal(), Beta(2, 2)),
    product_distribution(Beta(2, 2), Exponential()),
    product_distribution(m2, d2),
    product_distribution(m2, d2, m2, d2),
    # TODO: Vectors of distributions go to Distributions.Product rather than
    # Distributions.ProductDistribution, so this isn't implemented yet
    # ...

    # >1D arrays
    product_distribution(fill(Normal(), 2, 2)),
    product_distribution(fill(m2, 2, 2)),
    product_distribution(fill(d2, 2, 2)),
]

heterogeneous_products = [
    # These contain heterogeneous vectors, which means that the construction of the bijector
    # is type unstable. I don't think it's possible to fix this, but someone should probably at
    # least try.
    product_distribution([Normal() Beta(2, 2); Exponential() Uniform(-1, 1)]),
    product_distribution([m2 d2; m2 d2]),
]

@testset "Product distributions" begin
    for d in products
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end

    for d in heterogeneous_products
        VectorBijectors.test_all(
            d; expected_zero_allocs=(), test_construction_type_stable=false
        )
    end
end

end # module VBProductTests
