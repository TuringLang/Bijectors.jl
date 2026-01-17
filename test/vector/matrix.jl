module VBMatrixTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

# TODO(penelopeysm): ReverseDiff gives wrong results when differentiating
# through VecCorrBijector. Correctness tests are disabled for now.
# https://github.com/TuringLang/Bijectors.jl/issues/434
lkj_test_adtypes = [DI.AutoMooncake(), DI.AutoMooncakeForward()]

matrix_dists = [
    MatrixNormal(2, 4),
    MatrixNormal(3, 5),
    Wishart(7, Matrix{Float64}(I, 2, 2)),
    Wishart(7, Matrix{Float64}(I, 4, 4)),
    InverseWishart(7, Matrix{Float64}(I, 2, 2)),
    InverseWishart(7, Matrix{Float64}(I, 4, 4)),
    MatrixBeta(3, 3, 1000),
    MatrixBeta(5, 8, 1000),
    LKJ(3, 1.0),
    LKJ(7, 1.0),
]

@testset "Matrix distributions" begin
    for d in matrix_dists
        if d isa LKJ
            VectorBijectors.test_all(d; expected_zero_allocs=(), adtypes=lkj_test_adtypes)
        else
            VectorBijectors.test_all(d; expected_zero_allocs=())
        end
    end
end

end # module VBMatrixTests
