using Bijectors, LinearAlgebra, Test
using Bijectors: PDBijector, PDVecBijector

@testset "PDBijector" begin
    for d in [2, 5]
        b = PDBijector()
        dist = Wishart(d, Matrix{Float64}(I, d, d))
        x = rand(dist)
        # NOTE: `PDBijector` technically isn't bijective, and so the default `getjacobian`
        # used in the ChangesOfVariables.jl tests will fail as the jacobian will have determinant 0.
        # Hence, we disable those tests.
        test_bijector(b, x; test_not_identity=true, changes_of_variables_test=false)
    end
end

@testset "PDVecBijector" begin
    for d in [2, 5]
        b = PDVecBijector()
        dist = Wishart(d, Matrix{Float64}(I, d, d))
        x = rand(dist)
        y = b(x)

        # NOTE: `PDBijector` technically isn't bijective, and so the default `getjacobian`
        # used in the ChangesOfVariables.jl tests will fail as the jacobian will have determinant 0.
        # Hence, we disable those tests.
        test_bijector(b, x; test_not_identity=true, changes_of_variables_test=false)

        # Check that output sizes are computed correctly.
        tdist = transformed(dist, b)
        @test length(tdist) == length(y)
        @test tdist isa MultivariateDistribution

        dist_transformed = transformed(MvNormal(zeros(length(tdist)), I), inverse(b))
        @test size(dist_transformed) == size(x)
        @test dist_transformed isa MatrixDistribution
    end
end

@testset "triangular helpers on adjoint Cholesky factors" begin
    A = [2.0 1; 1 2]
    CL = cholesky(Hermitian(A, :L))
    @test Bijectors.upper_triangular(parent(CL.U)) == Matrix(CL.U)
    CU = cholesky(Hermitian(A, :U))
    @test Bijectors.lower_triangular(parent(CU.L)) == Matrix(CU.L)
end
