using Bijectors, DistributionsAD, LinearAlgebra, Test
using Bijectors: VecCorrBijector, CorrBijector

@testset "CorrBijector & VecCorrBijector" begin
    for d ∈ [1, 2, 5]
        b = CorrBijector()
        bvec = VecCorrBijector('C')

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
    end
end

@testset "VecCorrBijector on LKJCholesky" begin
    for d ∈ [2, 5]
        for dist in [LKJCholesky(d, 1, 'U'), LKJCholesky(d, 1, 'L')]
            b = bijector(dist)

            b_lkj = VecCorrBijector('C')
            x = rand(dist)
            y = b(x)
            y_lkj = b_lkj(x)

            @test y ≈ y_lkj

            binv = inverse(b)
            xinv = binv(y)
            binv_lkj = inverse(b_lkj)
            xinv_lkj = binv_lkj(y_lkj)

            @test xinv.U ≈ cholesky(xinv_lkj).U

            test_ad(x -> sum(b(binv(x))), y, (:Tracker,))

            # test_bijector is commented out for now, 
            # as isapprox is not defined for ::Cholesky types (the domain of LKJCholesky)
            # test_bijector(b, x; test_not_identity=d != 1, changes_of_variables_test=false)
        end
    end
end
