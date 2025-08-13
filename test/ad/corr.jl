@testset "AD for VecCorrBijector" begin
    @testset "d = $d" for d in (1, 2, 4)
        dist = LKJ(d, 2.0)
        b = bijector(dist)
        binv = inverse(b)

        x = rand(dist)
        y = b(x)

        # roundtrip
        test_ad(y -> sum(transform(b, binv(y))), y)
        # inverse only
        test_ad(y -> sum(transform(binv, y)), y)
    end
end

@testset "AD for VecCholeskyBijector" begin
    @testset "d = $d, uplo = $uplo" for d in (1, 2, 4), uplo in ('U', 'L')
        dist = LKJCholesky(d, 2.0, uplo)
        b = bijector(dist)
        binv = inverse(b)

        x = rand(dist)
        y = b(x)

        # roundtrip
        test_ad(y -> sum(transform(b, binv(y))), y)
        # inverse only
        test_ad(y -> sum(transform(binv, y)), y)
        # additionally check that cholesky_{upper,lower} is differentiable
        if uplo == 'U'
            test_ad(y -> sum(Bijectors.cholesky_upper(transform(b, y))), y)
        else
            test_ad(y -> sum(Bijectors.cholesky_lower(transform(b, y))), y)
        end
    end
end
