using Bijectors: rqs_params_from_raw, rqs_forward, rqs_univariate
using ForwardDiff: ForwardDiff

@testset "batched RQS parameters" begin
    @testset "T=$T, K=$K, D=$D, N=$N, B=$B" for T in (Float32, Float64),
        K in (4, 8), D in (1, 3), N in (1, 16),
        B in (2, 30)

        θ_raw = randn(T, (3K - 1) * D, N)
        widths, heights, derivatives = rqs_params_from_raw(θ_raw, D, B)

        @test size(widths) == (K + 1, D, N)
        @test size(heights) == (K + 1, D, N)
        @test size(derivatives) == (K + 1, D, N)

        # Raw parameters must not silently widen the element type.
        @test eltype(widths) == T
        @test eltype(heights) == T
        @test eltype(derivatives) == T

        for grid in (widths, heights)
            # The first knot is exactly -B (a prepended zero before the cumsum); the last is
            # B up to the floating-point error in the softmax normalisation.
            @test all(grid[1, :, :] .== -T(B))
            @test all(isapprox.(grid[end, :, :], T(B)))
            @test all(diff(grid; dims=1) .> 0)
        end

        @test all(derivatives[1, :, :] .== one(T))
        @test all(derivatives[end, :, :] .== one(T))
        @test all(derivatives .> 0)
    end
end

@testset "batched RQS forward" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8), D in (1, 3),
        N in (1, 8)

        B = 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        x = T(0.8B) .* (2 .* rand(T, D, N) .- 1)       # well inside [-B, B]
        y, logjac = rqs_forward(x, w, h, d)

        @test size(y) == (D, N)
        @test size(logjac) == (1, N)
        @test eltype(y) == T
        @test eltype(logjac) == T

        # Each column reproduces the legacy single-spline evaluation.
        for n in 1:N, i in 1:D
            @test y[i, n] ≈ rqs_univariate(w[:, i, n], h[:, i, n], d[:, i, n], x[i, n])
        end

        # logjac is the true log-derivative. The coupling is diagonal per dimension, so
        # exp(logjac[n]) equals the product over dims of dyᵢ/dxᵢ from ForwardDiff.
        for n in 1:N
            prod_dydx = one(T)
            for i in 1:D
                scalar_forward = function (xi)
                    xcol = reshape([j == i ? xi : x[j, n] for j in 1:D], D, 1)
                    return rqs_forward(xcol, w[:, :, n:n], h[:, :, n:n], d[:, :, n:n])[1][i]
                end
                dydx = ForwardDiff.derivative(scalar_forward, x[i, n])
                @test dydx > 0
                prod_dydx *= dydx
            end
            @test prod_dydx ≈ exp(logjac[1, n])
        end
    end

    @testset "out-of-range identity, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 2, 4
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        x = T[2B -2B 3B -3B; 2B -2B 3B -3B]           # all outside [-B, B]
        y, logjac = rqs_forward(x, w, h, d)
        @test y == x
        @test all(iszero, logjac)
    end
end
