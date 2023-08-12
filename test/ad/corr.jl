@testset "AD for VecCorrBijector" begin
    d = 4
    dist = LKJ(d, 2.0)
    b = bijector(dist)
    binv = inverse(b)

    x = rand(dist)
    y = b(x)

    test_ad(y) do x
        sum(transform(b, binv(x)))
    end

    test_ad(y) do y
        sum(transform(binv, y))
    end
end

@testset "AD for VecCholeskyBijector" begin
    d = 4
    dist = LKJCholesky(d, 2.0)
    b = bijector(dist)
    binv = inverse(b)

    x = rand(dist)
    y = b(x)

    test_ad(y) do y
        sum(transform(b, binv(y)))
    end

    test_ad(y) do y
        sum(Bijectors.cholesky_upper(transform(binv, y)))
    end
end
