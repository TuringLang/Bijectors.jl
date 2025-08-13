using Random: Xoshiro
using LinearAlgebra
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
        @testset "$mode" for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            Mooncake.TestUtils.test_rule(
                rng,
                Bijectors.find_alpha,
                x,
                y,
                z;
                is_primitive=true,
                perf_flag=:none,
                interp=Mooncake.MooncakeInterpreter(mode),
            )
            Mooncake.TestUtils.test_rule(
                rng,
                Bijectors.find_alpha,
                x,
                y,
                3;
                is_primitive=true,
                perf_flag=:none,
                interp=Mooncake.MooncakeInterpreter(mode),
            )
            Mooncake.TestUtils.test_rule(
                rng,
                Bijectors.find_alpha,
                x,
                y,
                UInt32(3);
                is_primitive=true,
                perf_flag=:none,
                interp=Mooncake.MooncakeInterpreter(mode),
            )
        end
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
    # Run multiple tests because we're working with `undef` entries, and so we
    # want to make sure that we hit cases where the `undef` entries have different values.
    # It's also just useful to test numerical stability for different realizations of `dist`.

    # NOTE(penelopeysm): https://github.com/TuringLang/Bijectors.jl/pull/357
    # changed the implementation of _link_chol_lkj... to improve its numerical stability.
    # The new implementation relies on the fact that the `LKJCholesky` distribution
    # yields samples for which each column is a unit vector. Naively using FiniteDifferences
    # to calculate a JVP (as ChainRulesTestUtils.test_rrule does) does not work, because
    # FD does not know about this constraint.
    # To solve this, we run the FD part of the test with the inputs projected onto a
    # subspace that has that constraint encoded. We have to then recover the original
    # output by un-projecting.
    # The PR linked above has a more detailed explanation.
    for i in 1:30
        dist = LKJCholesky(3, 4)
        rng = Xoshiro(i)
        spl = rand(rng, dist)

        @testset "_inv_link_chol_lkj" begin
            # This one doesn't need the fancy projection bits, so we can just
            # use test_rrule as usual.
            x = spl
            b = bijector(dist)
            y = b(x)
            test_rrule(
                _inv_link_chol_lkj_wrapper,
                y;
                testset_name="_inv_link_chol_lkj on $(typeof(x)) [$i]",
            )
        end

        # Set up a random tangent.
        ybar = rand(rng, 3) * 10
        fdm = FiniteDifferences.central_fdm(5, 1)

        # Functions to convert input to/from free parameters
        to_free_params(x::UpperTriangular) = [x[1, 2], x[1, 3], x[2, 3]]
        to_free_params(x::LowerTriangular) = [x[2, 1], x[3, 1], x[3, 2]]
        function from_x_free(x_free::AbstractVector, uplo::Symbol)
            x = UpperTriangular(zeros(eltype(x_free), 3, 3))
            x[1, 1] = 1
            x[1, 2] = x_free[1]
            x[1, 3] = x_free[2]
            x[2, 2] = sqrt(1 - x_free[1]^2)
            x[2, 3] = x_free[3]
            x[3, 3] = sqrt(1 - x_free[2]^2 - x_free[3]^2)
            return uplo == :U ? x : transpose(x)
        end
        # Function to reconvert the adjoint back into a triangular matrix
        function fd_xbar_to_cr_xbar(fd_xbar::AbstractVector, uplo::Symbol)
            x = UpperTriangular(zeros(eltype(fd_xbar), 3, 3))
            x[1, 2] = fd_xbar[1]
            x[1, 3] = fd_xbar[2]
            x[2, 3] = fd_xbar[3]
            return uplo == :U ? x : transpose(x)
        end

        @testset "_link_chol_lkj_from_upper" begin
            f = Bijectors._link_chol_lkj_from_upper
            x = spl.U

            # test primal is accurate
            y = f(x)
            cr_y, cr_pullback = ChainRulesCore.rrule(f, x)
            @test isapprox(y, cr_y)

            # test that the primal still works when going via free parameters
            f_via_free(x_free::AbstractVector) = f(from_x_free(x_free, :U))
            x_free = to_free_params(x)
            y_via_free = f_via_free(x_free)
            @test isapprox(y, y_via_free)

            # test pullback
            cr_xbar = cr_pullback(ybar)[2]
            fd_xbar = FiniteDifferences.j′vp(fdm, f_via_free, ybar, x_free)[1]
            @test isapprox(cr_xbar, fd_xbar_to_cr_xbar(fd_xbar, :U))
        end

        @testset "_link_chol_lkj_from_lower" begin
            f = Bijectors._link_chol_lkj_from_lower
            x = spl.L

            # test primal is accurate
            y = f(x)
            cr_y, cr_pullback = ChainRulesCore.rrule(f, x)
            @test isapprox(y, cr_y)

            # test that the primal still works when going via free parameters
            f_via_free(x_free::AbstractVector) = f(from_x_free(x_free, :L))
            x_free = to_free_params(x)
            y_via_free = f_via_free(x_free)
            @test isapprox(y, y_via_free)

            # test pullback
            cr_xbar = cr_pullback(ybar)[2]
            fd_xbar = FiniteDifferences.j′vp(fdm, f_via_free, ybar, x_free)[1]
            @test isapprox(cr_xbar, fd_xbar_to_cr_xbar(fd_xbar, :L))
        end
    end
end
