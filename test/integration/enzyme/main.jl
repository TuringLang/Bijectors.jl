using ADTypes
using Bijectors
using DifferentiationInterface
using Enzyme
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

BACKENDS = [
    (
        "EnzymeForward",
        AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    ),
    (
        "EnzymeReverse",
        AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
    ),
]

@testset "$backend" for (backend, adtype) in BACKENDS
    @info "Testing with backend: $backend"

    ENZYME_FWD_AND_1p11 =
        v"1.11" <= VERSION < v"1.12" && adtype isa AutoEnzyme{<:Enzyme.ForwardMode}
    ENZYME_RVS_AND_1p11 =
        v"1.11" <= VERSION < v"1.12" && adtype isa AutoEnzyme{<:Enzyme.ReverseMode}

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
            if d > 1 && ENZYME_FWD_AND_1p11
                # Segfaults
                # TODO: report
                @warn "Skipping forward-mode Enzyme for d=$d due to segfault"
                # test_ad(roundtrip, adtype, y)
                # test_ad(inverse_only, adtype, y)
            else
                test_ad(roundtrip, adtype, y)
                test_ad(inverse_only, adtype, y)
            end
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
        if ENZYME_FWD_AND_1p11
            @warn "Skipping forward-mode Enzyme for `g` on 1.11 due to segfault"
            # test_ad(g, adtype, randn(11))
        else
            test_ad(g, adtype, randn(11))
        end

        # logpdf of a flow with the inverse of a planar layer and two-dimensional inputs
        function finv(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), inverse(layer))
            x = θ[6:7]
            return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
        end
        if ENZYME_FWD_AND_1p11 || ENZYME_RVS_AND_1p11
            @test_throws Enzyme.LLVM.LLVMException test_ad(finv, adtype, randn(7))
        else
            test_ad(finv, adtype, randn(7))
        end

        function ginv(θ)
            layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
            flow = transformed(MvNormal(zeros(2), I), inverse(layer))
            x = reshape(θ[6:end], 2, :)
            return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
        end
        if ENZYME_FWD_AND_1p11
            @warn "Skipping forward-mode Enzyme for `ginv` on 1.11 due to segfault"
        elseif ENZYME_RVS_AND_1p11
            @test_throws Enzyme.LLVM.LLVMException test_ad(finv, adtype, randn(7))
        else
            test_ad(ginv, adtype, randn(11))
        end
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

        if ENZYME_FWD_AND_1p11
            @warn "Skipping forward-mode Enzyme for PDVecBijector due to segfaults on all instances"
            # test_ad(forward_only, adtype, vec(z))
            # test_ad(inverse_only, adtype, vec(z))
            # test_ad(inverse_chol_lower, adtype, y)
            # test_ad(inverse_chol_upper, adtype, y)
        else
            test_ad(forward_only, adtype, vec(z))
            test_ad(inverse_only, adtype, y)
            test_ad(inverse_chol_lower, adtype, y)
            test_ad(inverse_chol_upper, adtype, y)
        end
    end
end
