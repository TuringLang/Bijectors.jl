module VBReshapedTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using Enzyme: Enzyme
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
    vec(Poisson(3)),
    reshape(Poisson(3), (1, 1, 1, 1, 1)),
    reshape(MvNormal(zeros(2), I), (2, 1, 1)),
    reshape(MvNormal(zeros(4), I), (2, 2)),
    reshape(Dirichlet(ones(6)), (2, 3)),
    reshape(MatrixNormal(2, 4), 8),
    reshape(MatrixNormal(2, 5), 5, 2),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 16),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 1, 1, 4, 1, 4),
]

# Fails on 1.10: https://github.com/EnzymeAD/Enzyme.jl/issues/2987
adtypes_no_enz_rvs = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=EC.Forward, function_annotation=EC.Const),
]
reshaped_no_enzyme = [reshape(Beta(2, 2), (1, 1, 1, 1, 1))]

@testset "Reshaped distributions" begin
    for d in reshaped
        VectorBijectors.test_all(d; expected_zero_allocs=())
    end

    for d in reshaped_no_enzyme
        @static if VERSION >= v"1.11-"
            VectorBijectors.test_all(d; expected_zero_allocs=())
        else
            VectorBijectors.test_all(d; adtypes=adtypes_no_enz_rvs, expected_zero_allocs=())
        end
    end
end

end # module VBReshapedTests
