module VBProductTests

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

@testset "Product distributions" begin
    test_products_with(adtypes)
end

end # module VBProductTests
