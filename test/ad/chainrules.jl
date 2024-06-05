using ChainRulesTestUtils: ChainRulesCore

# HACK: This is a workaround to test `Bijectors._inv_link_chol_lkj` which produces an
# upper-triangular `Matrix`, leading to `test_rrule` comaring the _full_ `Matrix`,
# including the lower-triangular part which potentially contains `undef` entries.
# Here we simply wrap the rrule we want to test to also convert to PD form, thus
# avoiding any issues with the lower-triangular part.
function _inv_link_chol_lkj_wrapper(y)
    W, logJ = Bijectors._inv_link_chol_lkj(y)
    return Bijectors.pd_from_upper(W), logJ
end
function ChainRulesCore.rrule(::typeof(_inv_link_chol_lkj_wrapper), y::AbstractVector)
    (W, logJ), back = ChainRulesCore.rrule(Bijectors._inv_link_chol_lkj, y)
    X, back_X = ChainRulesCore.rrule(Bijectors.pd_from_upper, W)
    function pullback_inv_link_chol_lkj_wrapper((ΔX, ΔlogJ))
        (_, ΔW) = back_X(ChainRulesCore.unthunk(ΔX))
        (_, Δy) = back((ΔW, ΔlogJ))
        return (ChainRulesCore.NoTangent(), Δy)
    end
    return (X, logJ), pullback_inv_link_chol_lkj_wrapper
end

@testset "chainrules" begin
    x = randn()
    y = expm1(randn())
    z = randn()
    test_frule(Bijectors.find_alpha, x, y, z)
    test_rrule(Bijectors.find_alpha, x, y, z)

    test_rrule(
        Bijectors.combine,
        Bijectors.PartitionMask(3, [1], [2]) ⊢ ChainRulesTestUtils.NoTangent(),
        [1.0],
        [2.0],
        [3.0],
    )

    # ordered bijector
    b = Bijectors.OrderedBijector()
    test_rrule(Bijectors._transform_ordered, randn(5))
    test_rrule(Bijectors._transform_ordered, randn(5, 2))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5)))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5, 2)))

    # LKJ and LKJCholesky bijector
    dist = LKJCholesky(3, 4)
    # Run multiple tests because we're working with `undef` entries, and so we
    # want to make sure that we hit cases where the `undef` entries have different values.
    # It's also just useful to test numerical stability for different realizations of `dist`.
    for i in 1:30
        x = rand(dist)
        test_rrule(
            Bijectors._link_chol_lkj_from_upper,
            x.U,
            testset_name="_link_chol_lkj_from_upper on $(typeof(x)) [$i]"
        )
        test_rrule(
            Bijectors._link_chol_lkj_from_lower,
            x.L;
            testset_name="_link_chol_lkj_from_lower on $(typeof(x)) [$i]",
        )

        b = bijector(dist)
        y = b(x)

        test_rrule(
            _inv_link_chol_lkj_wrapper,
            y;
            testset_name="_inv_link_chol_lkj on $(typeof(x)) [$i]",
        )
    end
end
