using Enzyme: ForwardMode

@testset "PlanarLayer: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    ENZYME_FWD_AND_1p11 = VERSION >= v"1.11" && adtype isa AutoEnzyme{<:ForwardMode}

    # logpdf of a flow with a planar layer and two-dimensional inputs
    function f(θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = θ[6:7]
        return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
    end
    if ENZYME_FWD_AND_1p11
        # TODO: Report this upstream (or check if it's already been reported)
        @test_throws Enzyme.Compiler.EnzymeInternalError test_ad(f, adtype, randn(7))
    else
        test_ad(f, adtype, randn(7))
    end

    function g(θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end
    if ENZYME_FWD_AND_1p11
        @test_throws Enzyme.Compiler.EnzymeInternalError test_ad(g, adtype, randn(11))
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
    if ENZYME_FWD_AND_1p11
        @test_throws Enzyme.Compiler.EnzymeInternalError test_ad(f, adtype, randn(7))
    else
        test_ad(f, adtype, randn(7))
    end

    function ginv(θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), inverse(layer))
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end
    if ENZYME_FWD_AND_1p11
        @test_throws Enzyme.Compiler.EnzymeInternalError test_ad(g, adtype, randn(11))
    else
        test_ad(g, adtype, randn(11))
    end
end
