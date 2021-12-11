using Test

using Bijectors
using Bijectors: LeakyReLU

using LinearAlgebra
using ForwardDiff

true_logabsdetjac(b::Bijector{0}, x::Real) = (log ∘ abs)(ForwardDiff.derivative(b, x))
true_logabsdetjac(b::Bijector{0}, x::AbstractVector) = (log ∘ abs).(ForwardDiff.derivative.(b, x))
true_logabsdetjac(b::Bijector{1}, x::AbstractVector) = logabsdet(ForwardDiff.jacobian(b, x))[1]
true_logabsdetjac(b::Bijector{1}, xs::AbstractMatrix) = mapreduce(z -> true_logabsdetjac(b, z), vcat, eachcol(xs))

@testset "0-dim parameter, 0-dim input" begin
    b = LeakyReLU(0.1; dim=Val(0))
    x = 1.
    @test inverse(b)(b(x)) == x
    @test inverse(b)(b(-x)) == -x

    # Mixing of types
    # 1. Changes in input-type
    @assert eltype(b(Float32(1.))) == Float64
    @assert eltype(b(Float64(1.))) == Float64

    # 2. Changes in parameter-type
    b = LeakyReLU(Float32(0.1); dim=Val(0))
    @assert eltype(b(Float32(1.))) == Float32
    @assert eltype(b(Float64(1.))) == Float64

    # logabsdetjac
    @test logabsdetjac(b, x) == true_logabsdetjac(b, x)
    @test logabsdetjac(b, Float32(x)) == true_logabsdetjac(b, x)

    # Batch
    xs = randn(10)
    @test logabsdetjac(b, xs) == true_logabsdetjac(b, xs)
    @test logabsdetjac(b, Float32.(x)) == true_logabsdetjac(b, Float32.(x))

    @test logabsdetjac(b, -xs) == true_logabsdetjac(b, -xs)
    @test logabsdetjac(b, -Float32.(xs)) == true_logabsdetjac(b, -Float32.(xs))

    # Forward
    f = with_logabsdet_jacobian(b, xs)
    @test f[2] ≈ logabsdetjac(b, xs)
    @test f[1] ≈ b(xs)

    f = with_logabsdet_jacobian(b, Float32.(xs))
    @test f[2] == logabsdetjac(b, Float32.(xs))
    @test f[1] ≈ b(Float32.(xs))
end

@testset "0-dim parameter, 1-dim input" begin
    d = 2

    b = LeakyReLU(0.1; dim=Val(1))
    x = ones(d)
    @test inverse(b)(b(x)) == x
    @test inverse(b)(b(-x)) == -x

    # Batch
    xs = randn(d, 10)
    @test logabsdetjac(b, xs) == true_logabsdetjac(b, xs)
    @test logabsdetjac(b, Float32.(x)) == true_logabsdetjac(b, Float32.(x))

    @test logabsdetjac(b, -xs) == true_logabsdetjac(b, -xs)
    @test logabsdetjac(b, -Float32.(xs)) == true_logabsdetjac(b, -Float32.(xs))

    # Forward
    f = with_logabsdet_jacobian(b, xs)
    @test f[2] ≈ logabsdetjac(b, xs)
    @test f[1] ≈ b(xs)

    f = with_logabsdet_jacobian(b, Float32.(xs))
    @test f[2] == logabsdetjac(b, Float32.(xs))
    @test f[1] ≈ b(Float32.(xs))

    # Mixing of types
    # 1. Changes in input-type
    @assert eltype(b(ones(Float32, 2))) == Float64
    @assert eltype(b(ones(Float64, 2))) == Float64

    # 2. Changes in parameter-type
    b = LeakyReLU(Float32(0.1); dim=Val(1))
    @assert eltype(b(ones(Float32, 2))) == Float32
    @assert eltype(b(ones(Float64, 2))) == Float64
end
