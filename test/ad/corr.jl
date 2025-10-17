@testset "VecCorrBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    # Enzyme is tested separately as these tests are flaky
    # TODO(penelopeysm): Fix upstream and re-enable.
    if adtype isa AutoEnzyme
        continue
    end

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

@testset "VecCholeskyBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
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
