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

# Cover both `frule!!` and `rrule!!` paths registered in `BijectorsMooncakeExt`, across
# every `Base.IEEEFloat` width for the homogeneous case and a couple of integer types for
# the `Integer`-third-argument case.
@testset "Mooncake $mode: find_alpha" for mode in
                                          (Mooncake.ReverseMode, Mooncake.ForwardMode)
    rng = Xoshiro(123456)
    # Skip `Float16`: the root-finder in `find_alpha` doesn't keep enough precision for
    # Mooncake's default correctness tolerance.
    @testset "$P×$P×$P" for P in (Float32, Float64)
        x = P(randn(rng))
        y = P(expm1(randn(rng)))
        z = P(randn(rng))
        Mooncake.TestUtils.test_rule(
            rng,
            Bijectors.find_alpha,
            x,
            y,
            z;
            is_primitive=true,
            perf_flag=:none,
            mode=mode,
        )
    end
    @testset "Float64×Float64×$I" for I in (Int64, UInt32)
        x = randn(rng)
        y = expm1(randn(rng))
        z = I(3)
        Mooncake.TestUtils.test_rule(
            rng,
            Bijectors.find_alpha,
            x,
            y,
            z;
            is_primitive=true,
            perf_flag=:none,
            mode=mode,
        )
    end
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
