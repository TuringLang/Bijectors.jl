module VBReshapedTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

reshaped = [
    # 0-dim array output: doesn't work because
    # https://github.com/JuliaStats/Distributions.jl/issues/2025
    # reshape(Normal(), ()),
    vec(Normal()),
    reshape(Normal(), (1, 1, 1, 1, 1)),
    vec(Beta(2, 2)),
    reshape(Beta(2, 2), (1, 1, 1, 1, 1)),
    vec(Poisson(3)),
    reshape(Poisson(3), (1, 1, 1, 1, 1)),
    reshape(MvNormal(zeros(2), I), (2, 1, 1)),
    reshape(MvNormal(zeros(4), I), (2, 2)),
    reshape(Dirichlet(ones(6)), (2, 3)),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 16),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 1, 1, 4, 1, 4),
]

@testset "Reshaped distributions" begin
    for d in reshaped
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end
end

end # module VBReshapedTests
