module VBMatrixTests

import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using Test

include(joinpath(@__DIR__, "..", "shared", "vector_distributions.jl"))

# Enzyme is tested separately in test/integration/enzyme.
const adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
]

# ReverseDiff gives wrong results when differentiating through VecCorrBijector, so we run
# LKJ with Mooncake only. https://github.com/TuringLang/Bijectors.jl/issues/434
const lkj_adtypes = [DI.AutoMooncake(), DI.AutoMooncakeForward()]

@testset "Matrix distributions" begin
    test_matrix_dists_with(adtypes; lkj_adtypes=lkj_adtypes)
end

end # module VBMatrixTests
