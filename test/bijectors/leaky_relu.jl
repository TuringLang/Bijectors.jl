using Test

using Bijectors
using Bijectors: LeakyReLU

@testset "0-dim parameter, 0-dim input" begin
    b = LeakyReLU(0.1)

    # < 0
    x = -1.0
    test_bijector(b, x)

    # ≥ 0
    x = 1.0
    test_bijector(b, x; test_not_identity=false, test_types=true)

    # Float32
    b = LeakyReLU(Float32(b.α))

    # < 0
    x = -1.0f0
    test_bijector(b, x)

    # ≥ 0
    x = 1.0f0
    test_bijector(b, x; test_not_identity=false, test_types=true)
end

@testset "0-dim parameter, 1-dim input" begin
    d = 2
    b = LeakyReLU(0.1)

    # < 0
    x = -ones(d)
    test_bijector(b, x)

    # ≥ 0
    x = ones(d)
    test_bijector(b, x; test_not_identity=false)

    # Float32
    b = LeakyReLU(Float32.(b.α))
    # < 0
    x = -ones(Float32, d)
    test_bijector(b, x; test_types=true)

    # ≥ 0
    x = ones(Float32, d)
    test_bijector(b, x; test_not_identity=false, test_types=true)
end
