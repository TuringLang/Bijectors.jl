const order_base_dists = [
    Normal(),
    InverseGamma(2, 3),
    InverseGamma(2, 3) * -2,
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

function _gen_testcases(::Val{:order_orderstatistic})
    cases = VectorTestCase[]
    for d in order_base_dists
        unvec_only = (from_vec, from_linked_vec)
        push!(
            cases,
            VectorTestCase(
                "order statistic $(_case_name(d)) i=1 of n=10",
                OrderStatistic(d, 10, 1);
                expected_zero_allocs=unvec_only,
            ),
        )
        push!(
            cases,
            VectorTestCase(
                "order statistic $(_case_name(d)) i=10 of n=10",
                OrderStatistic(d, 10, 10);
                expected_zero_allocs=unvec_only,
            ),
        )
    end
    return cases
end

# JointOrderStatistics is only defined for continuous distributions. In the unlinked case
# the transform is identity. https://github.com/TuringLang/Bijectors.jl/issues/441 explains
# the unusually large `roundtrip_atol`.
function _gen_testcases(::Val{:order_joint})
    cases = VectorTestCase[]
    unlinked_only = (from_vec, to_vec)
    for d in order_base_dists
        d isa ContinuousUnivariateDistribution || continue
        push!(
            cases,
            VectorTestCase(
                "joint order statistic $(_case_name(d)) n=4 (all ranks)",
                JointOrderStatistics(d, 4);
                expected_zero_allocs=unlinked_only,
                roundtrip_atol=1e-1,
            ),
        )
        push!(
            cases,
            VectorTestCase(
                "joint order statistic $(_case_name(d)) n=10 ranks=2:5",
                JointOrderStatistics(d, 10, 2:5);
                expected_zero_allocs=unlinked_only,
                roundtrip_atol=1e-1,
            ),
        )
    end
    return cases
end

function _gen_testcases(::Val{:order_ordered})
    d = ordered(MvNormal([0.0, 1.0, 2.0], I))
    return [VectorTestCase("ordered MvNormal", d; expected_zero_allocs=(from_vec, to_vec))]
end
