module VBMultivariateTests

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

@testset "Multivariates" begin
    test_multivariates_with(adtypes)
end

end # module VBMultivariateTests
