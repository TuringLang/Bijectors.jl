using ADTypes
using Bijectors
using DifferentiationInterface
using Enzyme: Enzyme, set_runtime_activity, Forward, Reverse, Const
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences
using LinearAlgebra
using Test

const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))

function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
    @info "testing AD for function $f with $backend"
    ref_gradient = DifferentiationInterface.gradient(f, REF_BACKEND, x)
    gradient = DifferentiationInterface.gradient(f, backend, x)
    @test isapprox(gradient, ref_gradient; rtol=rtol, atol=atol)
end

const BACKENDS = [
    (
        "EnzymeForward",
        AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    ),
    (
        "EnzymeReverse",
        AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
    ),
]

# This entire test suite is broken on 1.11.
#
# https://github.com/EnzymeAD/Enzyme.jl/issues/2121
# https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
#
# The fix to this needs to be made in Julia itself: it seems that this has already been done
# in https://github.com/JuliaLang/llvm-project/pull/49 although whether this will be
# incorporated into the built Julia version itself seems unclear. See
# https://github.com/JuliaLang/julia/pull/59521#issuecomment-3300480633.
#
# If this does not end up being backported to 1.11, then we may have to permanently skip
# these tests.
#
# On another note: Ideally we'd use `@test_throws`. However, that doesn't work because
# `test_forward` itself calls `@test`, and the error is captured by that `@test`, not our
# `@test_throws`. Consequently `@test_throws` doesn't actually see any error. Weird Julia
# behaviour.
@static if VERSION < v"1.11"
    @testset "Enzyme: Bijectors.find_alpha" begin
        x = randn()
        y = expm1(randn())
        z = randn()

        @testset "forward" begin
            # No batches
            @testset for RT in (Const, Enzyme.Duplicated, Enzyme.DuplicatedNoNeed),
                Tx in (Const, Enzyme.Duplicated),
                Ty in (Const, Enzyme.Duplicated),
                Tz in (Const, Enzyme.Duplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            # Batches
            @testset for RT in
                          (Const, Enzyme.BatchDuplicated, Enzyme.BatchDuplicatedNoNeed),
                Tx in (Const, Enzyme.BatchDuplicated),
                Ty in (Const, Enzyme.BatchDuplicated),
                Tz in (Const, Enzyme.BatchDuplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end
        end
        @testset "reverse" begin
            # No batches
            @testset for RT in (Const, Enzyme.Active),
                Tx in (Const, Enzyme.Active),
                Ty in (Const, Enzyme.Active),
                Tz in (Const, Enzyme.Active)

                test_reverse(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            # TODO: Test batch mode
            # This is a bit problematic since Enzyme does not support all combinations of
            # activities currently
            # https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2480468728
        end
    end
end

@testset "$backend" for (backend, adtype) in BACKENDS
    @info "Testing with backend: $backend"

    @testset "VecCorrBijector" begin
        @info " - Testing VecCorrBijector"

        @testset "d = $d" for d in (1, 2, 4)
            @info "   - Dimension: $d"

            dist = LKJ(d, 2.0)
            b = bijector(dist)
            binv = inverse(b)

            x = rand(dist)
            y = b(x)

            roundtrip(y) = sum(transform(b, binv(y)))
            inverse_only(y) = sum(transform(binv, y))
            test_ad(roundtrip, adtype, y)
            test_ad(inverse_only, adtype, y)
        end
    end

    @testset "VecCholeskyBijector" begin
        @info " - Testing VecCholeskyBijector"

        @testset "d = $d, uplo = $uplo" for d in (1, 2, 4), uplo in ('U', 'L')
            @info "   - Dimension: $d, uplo: $uplo"

            dist = LKJCholesky(d, 2.0, uplo)
            b = bijector(dist)
            binv = inverse(b)

            x = rand(dist)
            y = b(x)
            cholesky_to_triangular =
                uplo == 'U' ? Bijectors.cholesky_upper : Bijectors.cholesky_lower

            roundtrip(y) = sum(transform(b, binv(y)))
            test_ad(roundtrip, adtype, y)

            # we need to tack on `cholesky_upper`/`cholesky_lower`, because directly calling
            # `sum` on a LinearAlgebra.Cholesky doesn't give a scalar
            inverse_only(y) = sum(cholesky_to_triangular(transform(binv, y)))
            test_ad(inverse_only, adtype, y)
        end
    end

    @testset "PlanarLayer" begin
        @info " - Testing PlanarLayer"
        # logpdf of a flow with a planar layer and two-dimensional inputs
        function f(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), layer)
            x = θ[6:7]
            return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
        end
        test_ad(f, adtype, randn(7))

        function g(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), layer)
            x = reshape(θ[6:end], 2, :)
            return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
        end
        test_ad(g, adtype, randn(11))

        # logpdf of a flow with the inverse of a planar layer and two-dimensional inputs
        function finv(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), inverse(layer))
            x = θ[6:7]
            return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
        end
        test_ad(finv, adtype, randn(7))

        function ginv(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), inverse(layer))
            x = reshape(θ[6:end], 2, :)
            return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
        end
        test_ad(ginv, adtype, randn(11))
    end

    @testset "PDVecBijector" begin
        @info " - Testing PDVecBijector"
        _topd(x) = x * x' + I

        d = 4
        b = Bijectors.PDVecBijector()
        binv = inverse(b)

        z = randn(d, d)
        x = _topd(z)
        y = b(x)

        forward_only(x) = sum(transform(b, _topd(reshape(x, d, d))))
        inverse_only(y) = sum(transform(binv, y))
        inverse_chol_lower(y) = sum(Bijectors.cholesky_lower(transform(binv, y)))
        inverse_chol_upper(y) = sum(Bijectors.cholesky_upper(transform(binv, y)))

        test_ad(forward_only, adtype, vec(z))
        test_ad(inverse_only, adtype, y)
        test_ad(inverse_chol_lower, adtype, y)
        test_ad(inverse_chol_upper, adtype, y)
    end
end
