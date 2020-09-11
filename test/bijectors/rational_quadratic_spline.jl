using Test
using Bijectors
using Bijectors: RationalQuadraticSpline

@setset "RationalQuadraticSpline" begin
    # Monotonic spline on '[-B, B]' with `K` intermediate knots/"connection points".
    K = 3; B = 2;
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
        @test b(x) ≠ x
        @test logabsdetjac(b, x) ≠ 0
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)
        x = [-x, x] # batch
        @test b(x) ≠ x
        @test b(x) isa Vector
        @test logabsdetjac(b, x) ≠ zeros(length(x))
        @test logabsdetjac(b, x) isa Vector
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)

        # Outside of domain
        x = 5.
        @test b(x) == x
        @test logabsdetjac(b, x) == 0
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)
        x = [-x, x] # batch
        @test b(x) == x
        @test b(x) isa Vector
        @test logabsdetjac(b, x) == zeros(length(x))
        @test logabsdetjac(b, x) isa Vector
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)

        # multivariate
        b = b_mv

        # Inside of domain
        x = [-0.5, 0.5]
        @test b(x) ≠ x
        @test b(x) isa Vector
        @test logabsdetjac(b, x) ≠ 0
        @test logabsdetjac(b, x) isa Real
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)
        x = hcat(-x, x) # batch
        @test b(x) ≠ x
        @test b(x) isa Matrix
        @test logabsdetjac(b, x) ≠ zeros(size(x, 2))
        @test logabsdetjac(b, x) isa Vector
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)

        # Outside of domain
        x = [-5., 5.]
        @test b(x) == x
        @test logabsdetjac(b, x) == 0
        @test logabsdetjac(b, x) isa Real
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)
        x = hcat(-x, x) # batch
        @test b(x) == x
        @test logabsdetjac(b, x) == zeros(size(x, 2))
        @test logabsdetjac(b, x) isa Vector
        res = forward(b, x)
        @test res.rv ≈ b(x)
        @test res.logabsdetjac ≈ logabsdetjac(b, x)
    end
end
