module VBMatrixTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
# Need ChainRules to differentiate through PDVecBijector, see ext/BijectorsReverseDiffExt.jl
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
using ChainRules: ChainRules

matrix_dists = [
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
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end
end

end # module VBMatrixTests
