"""
    LeakyReLU{T}(α::T) <: Bijector

Defines the invertible mapping

    x ↦ x if x ≥ 0 else αx

where α > 0.
"""
struct LeakyReLU{T} <: Bijector
    α::T
end

Functors.@functor LeakyReLU

inverse(b::LeakyReLU) = LeakyReLU(inv.(b.α))

function with_logabsdet_jacobian(b::LeakyReLU, x::Real)
    mask = x < zero(x)
    J = mask * b.α + !mask
    return J * x, log(abs(J))
end

# Array inputs.
function with_logabsdet_jacobian(b::LeakyReLU, x::AbstractArray)
    mask = x .< zero(eltype(x))
    J = mask .* b.α .+ (!).(mask)
    return J .* x, sum(log.(abs.(J)))
end

is_monotonically_increasing(::LeakyReLU) = true
