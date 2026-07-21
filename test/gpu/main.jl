using Bijectors
using Bijectors: BatchedRQS, transform, logabsdetjac, with_logabsdet_jacobian, inverse
using CUDA
using Random
using Test
using Zygote

# Scalar indexing on a GPU array is the usual sign that an operation fell back to a slow,
# element-by-element host loop. Disallowing it turns any such fallback into a test failure.
CUDA.allowscalar(false)

@testset "Batched RQS on CUDA" begin
    if !CUDA.functional()
        @info "CUDA is not functional on this agent, skipping GPU tests"
    else
        rng = MersenneTwister(1)
        K, D, N = 4, 3, 8
        B = 5.0f0
        n_raw = (3K - 1) * D

        θ_cpu = randn(rng, Float32, n_raw, N)
        x_cpu = randn(rng, Float32, D, N)
        θ_gpu = cu(θ_cpu)
        x_gpu = cu(x_cpu)

        @testset "runs on device and keeps Float32" begin
            b = BatchedRQS(θ_gpu, D, B)
            y = transform(b, x_gpu)
            lad = logabsdetjac(b, x_gpu)
            x_back = transform(inverse(b), y)
            @test y isa CuArray{Float32}
            @test lad isa CuArray{Float32}
            @test x_back isa CuArray{Float32}
            @test size(y) == (D, N)
            @test size(lad) == (N,)
        end

        @testset "device result matches host" begin
            b_cpu = BatchedRQS(θ_cpu, D, B)
            b_gpu = BatchedRQS(θ_gpu, D, B)
            y_cpu, lad_cpu = with_logabsdet_jacobian(b_cpu, x_cpu)
            y_gpu, lad_gpu = with_logabsdet_jacobian(b_gpu, x_gpu)
            @test Array(y_gpu) ≈ y_cpu rtol = 1.0f-4
            @test Array(lad_gpu) ≈ lad_cpu rtol = 1.0f-4

            xb_cpu = transform(inverse(b_cpu), y_cpu)
            xb_gpu = transform(inverse(b_gpu), y_gpu)
            @test Array(xb_gpu) ≈ xb_cpu rtol = 1.0f-4
            @test Array(xb_gpu) ≈ x_cpu rtol = 1.0f-4
        end

        @testset "gradient on device matches host" begin
            loss(θ, x) = sum(logabsdetjac(BatchedRQS(θ, D, B), x))
            g_cpu = only(Zygote.gradient(θ -> loss(θ, x_cpu), θ_cpu))
            g_gpu = only(Zygote.gradient(θ -> loss(θ, x_gpu), θ_gpu))
            @test g_gpu isa CuArray{Float32}
            @test Array(g_gpu) ≈ g_cpu rtol = 1.0f-3
        end
    end
end
