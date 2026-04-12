module VBCholeskyTests

using Distributions
using LinearAlgebra
using Test
using ADTypes: AutoEnzyme, AutoMooncake, AutoMooncakeForward, AutoReverseDiff
using Bijectors.VectorBijectors
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
using Enzyme: Enzyme, set_runtime_activity, Const, Forward, Reverse

# Need runtime activity for some reason.
# TODO(penelopeysm): Report upstream
const adtypes = [
    AutoReverseDiff(),
    AutoReverseDiff(; compile=true),
    AutoMooncake(),
    AutoMooncakeForward(),
    AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

dists = [
    LKJCholesky(1, 1.0, 'U'),
    LKJCholesky(1, 1.0, 'L'),
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

@testset "Cholesky" begin
    for d in dists
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end
end

end # module VBCholeskyTests
