module VBCholeskyTests

using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
# using Enzyme: Enzyme

# Need runtime activity for some reason.
# TODO(penelopeysm): Report upstream
const adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=EC.set_runtime_activity(EC.Forward), function_annotation=EC.Const),
    DI.AutoEnzyme(; mode=EC.set_runtime_activity(EC.Reverse), function_annotation=EC.Const),
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
