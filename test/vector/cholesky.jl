module VBCholeskyTests

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

@testset "Cholesky" begin
    test_cholesky_with(adtypes)
end

end # module VBCholeskyTests
