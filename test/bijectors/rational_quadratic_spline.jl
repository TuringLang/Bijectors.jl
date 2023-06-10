using Test
using Bijectors
using Bijectors: RationalQuadraticSpline
using LogExpFunctions

@testset "RationalQuadraticSpline" begin
    # Monotonic spline on '[-B, B]' with `K` intermediate knots/"connection points".
    d = 2
    K = 3
    B = 2
    b_uv = RationalQuadraticSpline(randn(K), randn(K), randn(K - 1), B)
    b_mv = RationalQuadraticSpline(randn(d, K), randn(d, K), randn(d, K - 1), B)

    @testset "Constructor" begin
        # univariate
        b = b_uv
        @test b.widths[1] ≈ -B
        @test b.widths[end] ≈ B
        @test b.widths[1] == minimum(b.widths)
        @test b.widths[end] == maximum(b.widths)
        @test b.heights[1] == minimum(b.heights)
        @test b.heights[end] == maximum(b.heights)
        @test all(b.derivatives .> 0)
        @test b.derivatives[1] == b.derivatives[end] == 1

        # multivariate
        b = b_mv
        @test all(b.widths[:, 1] .≈ -B)
        @test all(b.widths[:, end] .≈ B)
        @test all(b.widths[:, 1] .== minimum.(eachrow(b.widths)))
        @test all(b.widths[:, end] .== maximum.(eachrow(b.widths)))
        @test all(b.heights[:, 1] .== minimum.(eachrow(b.heights)))
        @test all(b.heights[end] .== maximum.(eachrow(b.heights)))
        @test all(all(b.derivatives .> 0))
        @test all(b.derivatives[:, 1] .== b.derivatives[:, end] .== 1)
    end

    @testset "Evaluation" begin
        # univariate
        b = b_uv

        # Inside of domain
        x = 0.5
        test_bijector(b, -x)
        test_bijector(b, x)

        # Outside of domain
        x = 5.0
        test_bijector(b, -x; y=-x, logjac=0)
        test_bijector(b, x; y=x, logjac=0)

        # multivariate
        b = b_mv

        # Inside of domain
        x = [-0.5, 0.5]
        test_bijector(b, x)

        # Outside of domain
        x = [-5.0, 5.0]
        test_bijector(b, x; y=x, logjac=zero(eltype(x)))
    end

    @testset "Float32 support" begin
        ws = randn(Float32, K)
        hs = randn(Float32, K)
        ds = randn(Float32, K - 1)

        Ws = randn(Float32, d, K)
        Hs = randn(Float32, d, K)
        Ds = randn(Float32, d, K - 1)

        # success of construction
        b = RationalQuadraticSpline(ws, hs, ds, B)
        bb = RationalQuadraticSpline(Ws, Hs, Ds, B)
    end

    @testset "consistency after commit" begin
        ws = randn(K)
        hs = randn(K)
        ds = randn(K - 1)

        Ws = randn(d, K)
        Hs = randn(d, K)
        Ds = randn(d, K - 1)

        Ws_t = hcat(zeros(size(Ws, 1)), LogExpFunctions.softmax(Ws; dims=2))
        Hs_t = hcat(zeros(size(Ws, 1)), LogExpFunctions.softmax(Hs; dims=2))

        # success of construction
        b = RationalQuadraticSpline(ws, hs, ds, B)
        b_mv = RationalQuadraticSpline(Ws, Hs, Ds, B)

        # consistency of evaluation
        @test all(
            (cumsum(vcat([zero(Float64)], LogExpFunctions.softmax(ws))) .- 0.5) * 2 * B .≈
            b.widths,
        )
        @test all(
            (cumsum(vcat([zero(Float64)], LogExpFunctions.softmax(hs))) .- 0.5) * 2 * B .≈
            b.heights,
        )
        @test all((2 * B) .* (cumsum(Ws_t; dims=2) .- 0.5) .≈ b_mv.widths)
        @test all((2 * B) .* (cumsum(Hs_t; dims=2) .- 0.5) .≈ b_mv.heights)
    end
end
