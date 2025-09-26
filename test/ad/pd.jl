_topd(x) = x * x' + I

@testset "PDVecBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    ENZYME_FWD_AND_1p11 = VERSION >= v"1.11" && adtype isa AutoEnzyme{<:Enzyme.ForwardMode}

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
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError test_ad(
            forward_only, adtype, vec(z)
        )
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError test_ad(
            inverse_only, adtype, vec(z)
        )
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError test_ad(
            inverse_chol_lower, adtype, y
        )
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError test_ad(
            inverse_chol_upper, adtype, y
        )
    else
        test_ad(forward_only, adtype, vec(z))
        test_ad(inverse_only, adtype, z)
        test_ad(inverse_chol_lower, adtype, y)
        test_ad(inverse_chol_upper, adtype, y)
    end
end
