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

function forward_batch(b::LeakyReLU, xs::Batch{<:AbstractVector})
    x = value(xs)
    
    J = let T = eltype(x), z = zero(T), o = one(T)
        @. (x < z) * b.α + (x > z) * o
    end
    return (result=Batch(J .* x), logabsdetjac=Batch(log.(abs.(J))))
end

# Array inputs.
function transform(b::LeakyReLU, x::AbstractArray)
    return let z = zero(eltype(x))
        @. (x < z) * b.α * x + (x > z) * x
    end
end

function logabsdetjac(b::LeakyReLU, x::AbstractArray)
    return sum(value(logabsdetjac_batch(b, Batch(x))))
end

function logabsdetjac_batch(b::LeakyReLU, xs::ArrayBatch{N}) where {N}
    x = value(xs)

    # Is really diagonal of jacobian
    J = let T = eltype(x), z = zero(T), o = one(T)
        @. (x < z) * b.α + (x > z) * o
    end
    
    logjac = if N ≤ 1
        sum(log ∘ abs, J)
    else
        vec(sum(map(log ∘ abs, J); dims = 1:N - 1))
    end

    return Batch(logjac)
end

# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU, x::AbstractArray)
    y, logjac = forward_batch(b, Batch(x))

    return (result = value(y), logabsdetjac = sum(value(logjac)))
end

function forward_batch(b::LeakyReLU, xs::ArrayBatch{N}) where {N}
    x = value(xs)
    
    # Is really diagonal of jacobian
    J = let T = eltype(x), z = zero(T), o = one(T)
        @. (x < z) * b.α + (x > z) * o
    end

    logjac = if N ≤ 1
        sum(log ∘ abs, J)
    else
        vec(sum(map(log ∘ abs, J); dims = 1:N - 1))
    end

    y = J .* x
    return (result=Batch(y), logabsdetjac=Batch(logjac))
end
