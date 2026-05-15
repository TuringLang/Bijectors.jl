using ADTypes
using Bijectors
using DifferentiationInterface
using Distributions
using FillArrays: Fill
using FiniteDifferences
using ForwardDiff: ForwardDiff
using LinearAlgebra
using Mooncake: Mooncake
using PDMats
using Random: Xoshiro
using Test

include(joinpath(@__DIR__, "..", "..", "test_resources.jl"))

const adtypes = [AutoMooncake(), AutoMooncakeForward()]

# ===== Mooncake rule for `find_alpha` =====

@testset "Mooncake $mode: find_alpha" for mode in (Mooncake.ReverseMode,)
    rng = Xoshiro(123456)
    x = randn()
    y = expm1(randn())
    z = randn()
    Mooncake.TestUtils.test_rule(
        rng, Bijectors.find_alpha, x, y, z; is_primitive=true, perf_flag=:none, mode=mode
    )
    Mooncake.TestUtils.test_rule(
        rng, Bijectors.find_alpha, x, y, 3; is_primitive=true, perf_flag=:none, mode=mode
    )
    Mooncake.TestUtils.test_rule(
        rng,
        Bijectors.find_alpha,
        x,
        y,
        UInt32(3);
        is_primitive=true,
        perf_flag=:none,
        mode=mode,
    )
end

@testset "Mooncake bijector AD" begin
    for c in generate_ad_testcases(), adtype in adtypes
        run_ad_case(c, adtype)
    end
end

@testset "Mooncake vector test_all" begin
    for c in generate_vector_testcases()
        run_vector_case(c, adtypes)
    end
end
