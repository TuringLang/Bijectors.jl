using Enzyme: ForwardMode

@testset "VecCorrBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    @testset "d = $d" for d in (1, 2, 4)
        dist = LKJ(d, 2.0)
        b = bijector(dist)
        binv = inverse(b)

        x = rand(dist)
        y = b(x)

        if adtype isa AutoEnzyme{<:ForwardMode} && d == 1
            # For d == 1, y has length 0, and DI doesn't handle this well
            # https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/802
            @test_throws DivideError test_ad(y -> sum(transform(b, binv(y))), adtype, y)
            @test_throws DivideError test_ad(y -> sum(transform(binv, y)), adtype, y)
        else
            # roundtrip
            test_ad(y -> sum(transform(b, binv(y))), adtype, y)
            # inverse only
            test_ad(y -> sum(transform(binv, y)), adtype, y)
        end
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

        # roundtrip
        test_ad(y -> sum(transform(b, binv(y))), adtype, y)
        # inverse (we need to tack on `cholesky_upper`/`cholesky_lower`,
        # because directly calling `sum` on a LinearAlgebra.Cholesky doesn't
        # give a scalar)
        test_ad(y -> sum(cholesky_to_triangular(transform(binv, y))), adtype, y)
    end
end
