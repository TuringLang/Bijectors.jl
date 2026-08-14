using Bijectors: rqs_params_from_raw, rqs_forward, rqs_inverse, rqs_univariate, BatchedRQS
using Bijectors: transform, with_logabsdet_jacobian, logabsdetjac, inverse
using ForwardDiff: ForwardDiff

@testset "batched RQS parameters" begin
    @testset "T=$T, K=$K, D=$D, N=$N, B=$B" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 16),
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
            # Both boundary knots are pinned exactly.
            @test all(grid[1, :, :] .== -T(B))
            @test all(grid[end, :, :] .== T(B))
            @test all(diff(grid; dims=1) .> 0)
        end

        @test all(derivatives[1, :, :] .== one(T))
        @test all(derivatives[end, :, :] .== one(T))
        @test all(derivatives .> 0)
    end

    @testset "raw parameter layout" begin
        K, D, N, B = 4, 2, 3, 5.0
        base = zeros((3K - 1) * D, N)
        w0, h0, d0 = rqs_params_from_raw(base, D, B)
        for (idx, moved) in ((1, :widths), (K + 1, :heights), (2K + 1, :derivatives))
            θ = copy(base)
            θ[idx, 1] += 1.0
            w, h, d = rqs_params_from_raw(θ, D, B)
            @test (w != w0) == (moved == :widths)
            @test (h != h0) == (moved == :heights)
            @test (d != d0) == (moved == :derivatives)
        end
    end

    @testset "extreme logits stay finite, T=$T" for T in (Float32, Float64)
        # Without the minimum bin fraction and minimum derivative these logits underflow a
        # width, a height, and a slope to exactly zero and produce NaN from in-range inputs.
        K, D, N, B = 4, 1, 1, 5
        θ = zeros(T, 3K - 1, N)
        θ[1, 1] = 110
        θ[K + 1, 1] = 110
        θ[2K + 1, 1] = -150
        w, h, d = rqs_params_from_raw(θ, D, B)

        @test all(diff(w; dims=1) .>= 2 * T(B) * T(1e-3) * T(0.99))
        @test all(diff(h; dims=1) .>= 2 * T(B) * T(1e-3) * T(0.99))
        @test all(d .>= T(1e-3))

        # Probe interior points, every knot, and out-of-range points.
        xs = reshape(vcat(T[-0.99B, 0, 0.99B, -B - 1, B + 1], w[:, 1, 1]), 1, :)
        wN = repeat(w; outer=(1, 1, size(xs, 2)))
        hN = repeat(h; outer=(1, 1, size(xs, 2)))
        dN = repeat(d; outer=(1, 1, size(xs, 2)))

        y, logjac = rqs_forward(xs, wN, hN, dN)
        @test all(isfinite, y)
        @test all(isfinite, logjac)
        xb, logjac_inv = rqs_inverse(y, wN, hN, dN)
        @test all(isfinite, xb)
        @test all(isfinite, logjac_inv)

        gx = ForwardDiff.gradient(
            v -> sum(rqs_forward(reshape(v, 1, :), wN, hN, dN)[1]), vec(xs)
        )
        @test all(isfinite, gx)
        gθ = ForwardDiff.gradient(vec(θ)) do v
            wg, hg, dg = rqs_params_from_raw(reshape(v, :, N), D, B)
            x1 = reshape(T[0.99B], 1, 1)
            sum(rqs_forward(x1, wg, hg, dg)[2])
        end
        @test all(isfinite, gθ)
    end
end

@testset "batched RQS forward" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
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

@testset "batched RQS inverse" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 8)

        B = 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        rtol = T == Float32 ? 1.0f-4 : 1.0e-9

        # inverse ∘ forward
        x = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        y, logjac_fwd = rqs_forward(x, w, h, d)
        xback, logjac_inv = rqs_inverse(y, w, h, d)
        @test xback ≈ x rtol = rtol
        @test logjac_inv ≈ -logjac_fwd rtol = rtol

        # forward ∘ inverse (heights are also constrained to [-B, B])
        yin = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        xr, _ = rqs_inverse(yin, w, h, d)
        yr, _ = rqs_forward(xr, w, h, d)
        @test yr ≈ yin rtol = rtol
    end

    @testset "out-of-range identity, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 2, 4
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        y = T[2B -2B 3B -3B; 2B -2B 3B -3B]
        x, logjac = rqs_inverse(y, w, h, d)
        @test x == y
        @test all(iszero, logjac)
    end

    @testset "boundary gradient is finite, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 1, 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        # points spanning below, on, and above the range
        y = reshape(T[-B - 1, -B, 0, B, B + 1], D, N)
        g = ForwardDiff.gradient(
            v -> sum(rqs_inverse(reshape(v, D, N), w, h, d)[1]), vec(y)
        )
        @test all(isfinite, g)
        gj = ForwardDiff.gradient(
            v -> sum(rqs_inverse(reshape(v, D, N), w, h, d)[2]), vec(y)
        )
        @test all(isfinite, gj)
    end
end

@testset "BatchedRQS bijector" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 8)

        B = 5
        θ_raw = randn(T, (3K - 1) * D, N)
        w, h, d = rqs_params_from_raw(θ_raw, D, B)
        b = BatchedRQS(w, h, d)
        x = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        rtol = T == Float32 ? 1.0f-4 : 1.0e-9

        # The convenience constructor matches building from constrained params.
        b2 = BatchedRQS(θ_raw, D, B)
        @test b2.widths == w && b2.heights == h && b2.derivatives == d

        # transform / logabsdetjac / with_logabsdet_jacobian are consistent.
        y = transform(b, x)
        ladj = logabsdetjac(b, x)
        y2, ladj2 = with_logabsdet_jacobian(b, x)
        @test y == y2
        @test ladj == ladj2
        @test size(y) == (D, N)
        @test size(ladj) == (N,)
        @test eltype(y) == T
        @test eltype(ladj) == T

        # Inverse via the generic wrapper round-trips and negates the log-det.
        xback, ladj_inv = with_logabsdet_jacobian(inverse(b), y)
        @test xback ≈ x rtol = rtol
        @test ladj_inv ≈ -ladj rtol = rtol
        @test transform(inverse(b), y) ≈ x rtol = rtol

        # Type stability.
        @inferred transform(b, x)
        @inferred with_logabsdet_jacobian(b, x)
    end

    @testset "shape mismatch is rejected" begin
        w = randn(Float64, 5, 2, 3)
        @test_throws DimensionMismatch BatchedRQS(w, w, randn(Float64, 5, 2, 4))
    end
end
