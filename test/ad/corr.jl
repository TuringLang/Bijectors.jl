@testset "VecCorrBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    @testset "d = $d" for d in (1, 2, 4)
        dist = LKJ(d, 2.0)
        b = bijector(dist)
        binv = inverse(b)

        x = rand(dist)
        y = b(x)

        # roundtrip
        test_ad(y -> sum(transform(b, binv(y))), adtype, y)
        # inverse only
        test_ad(y -> sum(transform(binv, y)), adtype, y)
    end
end

@testset "VecCholeskyBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    @testset "d = $d, uplo = $uplo" for d in (1, 2, 4), uplo in ('U', 'L')
        dist = LKJCholesky(d, 2.0, uplo)
        b = bijector(dist)
        binv = inverse(b)

        x = rand(dist)
        y = b(x)

        # roundtrip
        test_ad(y -> sum(transform(b, binv(y))), adtype, y)
        # inverse (we need to tack on `cholesky_upper`/`cholesky_lower`,
        # because directly calling `sum` on a LinearAlgebra.Cholesky doesn't
        # give a scalar)
        if uplo == 'U'
            test_ad(y -> sum(Bijectors.cholesky_upper(transform(binv, y))), adtype, y)
        else
            test_ad(y -> sum(Bijectors.cholesky_lower(transform(binv, y))), adtype, y)
        end
    end
end
