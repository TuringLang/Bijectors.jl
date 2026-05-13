# Parameterised AD tests for specific bijectors. Each function takes an `adtype` and runs
# the test body once for that backend. Callers in `test/ad/*.jl` and
# `test/integration/enzyme/main.jl` loop over their own adtype lists and invoke these.
#
# Requires `test_ad` (from ad_test_utils.jl) and `Bijectors`/`Distributions` to be loaded.

using Bijectors
using Distributions
using LinearAlgebra

function test_veccorrbijector_ad(adtype)
    @testset "d = $d" for d in (1, 2, 4)
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

function test_veccholeskybijector_ad(adtype)
    @testset "d = $d, uplo = $uplo" for d in (1, 2, 4), uplo in ('U', 'L')
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

function test_planarlayer_ad(adtype)
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

function test_pdvecbijector_ad(adtype)
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

function test_stackedbijector_ad(adtype)
    dist1 = Dirichlet(4, 1.0)
    b1 = bijector(dist1)

    dist2 = LogNormal(0.0, 1.0)
    b2 = bijector(dist2)

    x1 = rand(dist1)
    x2 = rand(dist2)

    y1 = b1(x1)
    y2 = b2(x2)

    b = Stacked((b1, b2), (1:4, 5:5))
    binv = inverse(b)

    y = vcat(y1, [y2])
    x = binv(y)

    test_ad(y -> sum(transform(b, binv(y))), adtype, y)
    test_ad(y -> sum(transform(binv, y)), adtype, y)

    bvec = Stacked([b1, b2], [1:4, 5:5])
    bvec_inv = inverse(bvec)

    test_ad(y -> sum(transform(bvec, binv(y))), adtype, y)
    test_ad(y -> sum(transform(bvec_inv, y)), adtype, y)
end
