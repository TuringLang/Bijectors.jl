using Bijectors: rqs_params_from_raw

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
