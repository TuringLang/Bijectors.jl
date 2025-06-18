using Random: Xoshiro
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

    if @isdefined Mooncake
        rng = Xoshiro(123456)
        Mooncake.TestUtils.test_rule(
            rng,
            Bijectors.find_alpha,
            x,
            y,
            z;
            is_primitive=true,
            perf_flag=:none,
            interp=Mooncake.MooncakeInterpreter(),
        )
        Mooncake.TestUtils.test_rule(
            rng,
            Bijectors.find_alpha,
            x,
            y,
            3;
            is_primitive=true,
            perf_flag=:none,
            interp=Mooncake.MooncakeInterpreter(),
        )
        Mooncake.TestUtils.test_rule(
            rng,
            Bijectors.find_alpha,
            x,
            y,
            UInt32(3);
            is_primitive=true,
            perf_flag=:none,
            interp=Mooncake.MooncakeInterpreter(),
        )
    end

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
        # Note (penelopeysm): The reimplementation of _link_chol_lkj... in
        # https://github.com/TuringLang/Bijectors.jl/pull/357 improves its
        # numerical stability. However, it relies on the fact that each column
        # of the input matrix is a unit vector, which we cannot express in
        # code. This messes very thoroughly with FiniteDifferences, which is
        # what ChainRulesTestUtils uses to test the numerical accuracy of the
        # rrule.
        # To get around this, we specify the output tangent used as the input
        # to the rrule, and scale it down to be close to zero. This helps to
        # mitigate the numerical issues with FiniteDifferences. We also need to
        # set an abnormally large atol (the default is 1e-9). Not ideal, but in
        # general there is no way to get this to work nicely because there is
        # no way to tell FiniteDifferences about the constraints on the input
        # matrix.
        # In practice, I am hoping that it's not a problem because AD isn't
        # being run directly on the transformation (logabsdetjac directly
        # returns the relevant term) and the implementation of logabsdetjac
        # isn't touched.
        rng = Xoshiro(i)
        scale = 10
        x = rand(rng, dist)
        test_rrule(
            Bijectors._link_chol_lkj_from_upper,
            x.U;
            testset_name="_link_chol_lkj_from_upper on $(typeof(x)) [$i]",
            output_tangent=rand(rng, 3) / scale,
            atol=0.15,
        )
        test_rrule(
            Bijectors._link_chol_lkj_from_lower,
            x.L;
            testset_name="_link_chol_lkj_from_lower on $(typeof(x)) [$i]",
            output_tangent=rand(rng, 3) / scale,
            atol=0.15,
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
