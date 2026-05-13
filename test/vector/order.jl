module VBOrderTests

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

# ReverseDiff can't differentiate through JointOrderStatistics because of the heavy
# setindex! usage. https://github.com/JuliaDiff/ReverseDiff.jl/issues/43
const joint_adtypes = [DI.AutoMooncake(), DI.AutoMooncakeForward()]

@testset "Order statistics" begin
    test_order_with(adtypes; joint_adtypes=joint_adtypes)
end

end # module VBOrderTests
