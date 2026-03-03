module VBCholeskyTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
using Enzyme: Enzyme, set_runtime_activity, Const, Forward, Reverse

# Need runtime activity for some reason.
# TODO(penelopeysm): Report upstream
const adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    DI.AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

dists = [
    # Note: can't test LKJCholesky(1, ...) because its linked vector is length-zero and
    # DifferentiationInterface trips up with empty vectors.
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
