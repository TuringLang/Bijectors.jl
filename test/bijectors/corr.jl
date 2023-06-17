using Bijectors, DistributionsAD, LinearAlgebra, Test
using Bijectors: VecCorrBijector, VecCholeskyBijector, CorrBijector

@testset "CorrBijector & VecCorrBijector" begin
    for d in [1, 2, 5]
        b = CorrBijector()
        bvec = VecCorrBijector()

        dist = LKJ(d, 1)
        x = rand(dist)

        y = b(x)
        yvec = bvec(x)

        # Make sure that they represent the same thing.
        @test Bijectors.triu1_to_vec(y) ≈ yvec

        # Check the inverse.
        binv = inverse(b)
        xinv = binv(y)
        bvecinv = inverse(bvec)
        xvecinv = bvecinv(yvec)

        @test xinv ≈ xvecinv

        # And finally that the `logabsdetjac` is the same.
        @test logabsdetjac(bvec, x) ≈ logabsdetjac(b, x)

        # NOTE: `CorrBijector` technically isn't bijective, and so the default `getjacobian`
        # used in the ChangesOfVariables.jl tests will fail as the jacobian will have determinant 0.
        # Hence, we disable those tests.
        test_bijector(b, x; test_not_identity=d != 1, changes_of_variables_test=false)
        test_bijector(bvec, x; test_not_identity=d != 1, changes_of_variables_test=false)

        test_ad(x -> sum(bvec(bvecinv(x))), yvec)

        # Check that output sizes are computed correctly.
        dist = transformed(dist)
        @test length(dist) == length(yvec)

        dist_unconstrained = transformed(MvNormal(zeros(length(dist)), I), inverse(bvec))
        @test size(dist_unconstrained) == size(x)
    end
end

@testset "VecCholeskyBijector" begin
    for d in [2, 5]
        for dist in [LKJCholesky(d, 1, 'U'), LKJCholesky(d, 1, 'L')]
            b = bijector(dist)

            b_lkj = VecCorrBijector()
            x = rand(dist)
            y = b(x)
            y_lkj = b_lkj(x)

            @test y ≈ y_lkj

            binv = inverse(b)
            xinv = binv(y)
            binv_lkj = inverse(b_lkj)
            xinv_lkj = binv_lkj(y_lkj)

            @test xinv.U ≈ cholesky(xinv_lkj).U

            test_ad(x -> sum(b(binv(x))), y)

            # test_bijector is commented out for now, 
            # as isapprox is not defined for ::Cholesky types (the domain of LKJCholesky)
            # test_bijector(b, x; test_not_identity=d != 1, changes_of_variables_test=false)
        end
    end
end
