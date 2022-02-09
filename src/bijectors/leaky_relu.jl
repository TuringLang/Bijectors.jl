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

Base.inv(b::LeakyReLU) = LeakyReLU(inv.(b.α))

# (N=0) Univariate case
function transform(b::LeakyReLU, x::Real)
    mask = x < zero(x)
    return mask * b.α * x + !mask * x
end

function logabsdetjac(b::LeakyReLU, x::Real)
    mask = x < zero(x)
    J = mask * b.α + (1 - mask) * one(x)
    return log(abs(J))
end

# We implement `with_logabsdet_jacobian` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU, x::Real)
    mask = x < zero(x)
    J = mask * b.α + !mask * one(x)
    return (result=J * x, logabsdetjac=log(abs(J)))
end

# Array inputs.
function transform(b::LeakyReLU, x::AbstractArray)
    return let z = zero(eltype(x))
        @. (x < z) * b.α * x + (x > z) * x
    end
end

# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU, x::AbstractArray)
    y, logjac = forward_batch(b, Batch(x))

    return (result = value(y), logabsdetjac = sum(value(logjac)))
end
