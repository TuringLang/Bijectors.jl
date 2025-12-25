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

matrix_dists = [Wishart(7, [1 0; 0 1]), InverseWishart(7, [1 0; 0 1])]

@testset "Matrix distributions" begin
    for d in matrix_dists
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end
end

end # module VBMatrixTests
