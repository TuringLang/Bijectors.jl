module VBOrderTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

base_dists = [
    Normal(),
    InverseGamma(2, 3),
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

@testset "Order statistics" begin
    for d in base_dists
        unvec_only = (from_vec, from_linked_vec)
        VectorBijectors.test_all(OrderStatistic(d, 10, 1); expected_zero_allocs=unvec_only)
        VectorBijectors.test_all(OrderStatistic(d, 10, 10); expected_zero_allocs=unvec_only)
        # JointOrderStatistics is only defined for continuous distributions (technically, it
        # *should* work for discrete distributions whose support is some set which has a
        # total order, but Distributions.jl doesn't actually implement that).
        if d isa ContinuousUnivariateDistribution
            # These may avoid allocations if the transform is identity
            all = (from_vec, to_vec, from_linked_vec, to_linked_vec)
            vec_only = (from_vec, to_vec)
            zero_allocs = d isa Normal ? all : vec_only
            VectorBijectors.test_all(
                JointOrderStatistics(d, 10); expected_zero_allocs=zero_allocs
            )
            VectorBijectors.test_all(
                JointOrderStatistics(d, 10, 2:5); expected_zero_allocs=zero_allocs
            )
        end
    end
end

end # module VBOrderTests
