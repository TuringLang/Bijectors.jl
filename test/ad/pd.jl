_topd(x) = x * x' + I

@testset "PDVecBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    # Enzyme is tested separately as these tests are flaky
    # TODO(penelopeysm): Fix upstream and re-enable.
    if adtype isa AutoEnzyme
        continue
    end

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
