module BijectorsMooncakeTests

using Bijectors: Bijectors
using Mooncake: Mooncake
using Random: Xoshiro
using Test

x = randn()
y = expm1(randn())
z = randn()
rng = Xoshiro(123456)

# TODO: Enable Mooncake.ForwardMode as well.
@testset "Mooncake $mode: find_alpha" for mode in (Mooncake.ReverseMode,)
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
    Mooncake.TestUtils.test_rule(
        rng,
        Bijectors.find_alpha,
        x,
        y,
        3;
        is_primitive=true,
        perf_flag=:none,
        mode=mode,
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

end # module
