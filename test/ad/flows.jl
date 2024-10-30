@testset "PlanarLayer" begin
    # TODO(mhauru) Remove the EnzymeReverseCrash marks once this has been fixed:
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2029
    # TODO(mhauru) Remove the EnzymeReverse marks once this has been fixed:
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2030
    # logpdf of a flow with a planar layer and two-dimensional inputs
    test_ad(randn(7), (:EnzymeReverseCrash,)) do θ
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = θ[6:7]
        return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
    end
    test_ad(randn(11), (:EnzymeReverse,)) do θ
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end

    # logpdf of a flow with the inverse of a planar layer and two-dimensional inputs
    test_ad(randn(7), (:EnzymeReverseCrash,)) do θ
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), inverse(layer))
        x = θ[6:7]
        return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
    end
    test_ad(randn(11), (:EnzymeReverse,)) do θ
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), inverse(layer))
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end
end
