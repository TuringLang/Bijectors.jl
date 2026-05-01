module VBOrderTests

using ADTypes
using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using Bijectors: ordered
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

base_dists = [
    Normal(),
    InverseGamma(2, 3),
    InverseGamma(2, 3) * -2,
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

# TODO(penelopeysm): ReverseDiff can't differentiate through JointOrderStatistics transform
# because of the heavy setindex! usage.
# https://github.com/JuliaDiff/ReverseDiff.jl/issues/43 We just avoid testing it for now.
joint_test_adtypes = [
    AutoMooncake(),
    AutoMooncakeForward(),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
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
            # In the unlinked case, the transform is identity.
            #
            # See https://github.com/TuringLang/Bijectors.jl/issues/441 for details about
            # the unusually large atol.
            unlinked_only = (from_vec, to_vec)
            VectorBijectors.test_all(
                JointOrderStatistics(d, 4);
                expected_zero_allocs=unlinked_only,
                adtypes=joint_test_adtypes,
                roundtrip_atol=1e-1,
            )
            VectorBijectors.test_all(
                JointOrderStatistics(d, 10, 2:5);
                expected_zero_allocs=unlinked_only,
                adtypes=joint_test_adtypes,
                roundtrip_atol=1e-1,
            )
        end
    end

    # Bijectors.ordered
    VectorBijectors.test_all(
        ordered(MvNormal([0.0, 1.0, 2.0], I)); expected_zero_allocs=(from_vec, to_vec)
    )
end

end # module VBOrderTests
