_topd(x) = x * x' + I

@testset "PDVecBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    d = 4
    b = Bijectors.PDVecBijector()
    binv = inverse(b)

    z = randn(d, d)
    x = _topd(z)
    y = b(x)

    test_ad(x -> sum(transform(b, _topd(reshape(x, d, d)))), adtype, vec(z))
    test_ad(y -> sum(transform(binv, y)), adtype, y)
    test_ad(y -> sum(Bijectors.cholesky_lower(transform(binv, y))), adtype, y)
    test_ad(y -> sum(Bijectors.cholesky_upper(transform(binv, y))), adtype, y)
end
