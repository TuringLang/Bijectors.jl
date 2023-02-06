using Bijectors, DistributionsAD, LinearAlgebra, Test
using Bijectors: VecCorrBijector, CorrBijector

@testset "PDBijector" begin
    d = 3

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
    test_bijector(b, x; test_not_identity=true, changes_of_variables_test=false)
    test_bijector(bvec, x; test_not_identity=true, changes_of_variables_test=false)
end
