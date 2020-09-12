"""
    LeakyReLU{T, N}(α::T) <: Bijector{N}

Defines the invertible mapping

    x ↦ x if x ≥ 0 else αx

where α > 0.
"""
struct LeakyReLU{T, N} <: Bijector{N}
    α::T
end

LeakyReLU(α::T; dim::Val{N} = Val(0)) where {T<:Real, N} = LeakyReLU{T, N}(α)
LeakyReLU(α::T; dim::Val{N} = Val(D)) where {D, T<:AbstractArray{<:Real, D}, N} = LeakyReLU{T, N}(α)

up1(b::LeakyReLU{T, N}) where {T, N} = LeakyReLU{T, N + 1}(b.α)

# (N=0) Univariate case
function (b::LeakyReLU{<:Any, 0})(x::Real)
    mask = x < zero(x)
    return mask * b.α * x + !mask * x
end
(b::LeakyReLU{<:Any, 0})(x::AbstractVector{<:Real}) = map(b, x)

function Base.inv(b::LeakyReLU{<:Any,N}) where N
    invα = inv.(b.α)
    return LeakyReLU{typeof(invα),N}(invα)
end

function logabsdetjac(b::LeakyReLU{<:Any, 0}, x::Real)
    mask = x < zero(x)
    J = mask * b.α + (1 - mask) * one(x)
    return log(abs(J))
end
logabsdetjac(b::LeakyReLU{<:Real, 0}, x::AbstractVector{<:Real}) = map(x -> logabsdetjac(b, x), x)


# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{<:Any, 0}, x::Real)
    mask = x < zero(x)
    J = mask * b.α + !mask * one(x)
    return (rv=J * x, logabsdetjac=log(abs(J)))
end

# Batched version
function forward(b::LeakyReLU{<:Any, 0}, x::AbstractVector)
    J = let z = zero(x), o = one(x)
        @. (x < z) * b.α + (x > z) * o
    end
    return (rv=J .* x, logabsdetjac=log.(abs.(J)))
end

# (N=1) Multivariate case
function (b::LeakyReLU{<:Any, 1})(x::AbstractVecOrMat)
    return let z = zero(x)
        @. (x < z) * b.α * x + (x > z) * x
    end
end

function logabsdetjac(b::LeakyReLU{<:Any, 1}, x::AbstractVecOrMat)
    # Is really diagonal of jacobian
    J = let z = zero(x), o = one(x)
        @. (x < z) * b.α + (x > z) * o
    end

    if x isa AbstractVector
        return sum(log.(abs.(J)))
    elseif x isa AbstractMatrix
        return vec(sum(log.(abs.(J)); dims = 1))  # sum along column
    end
end

# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{<:Any, 1}, x::AbstractVecOrMat)
    # Is really diagonal of jacobian
    J = let z = zero(x), o = one(x)
        @. (x < z) * b.α + (x > z) * o
    end

    if x isa AbstractVector
        logjac = sum(log.(abs.(J)))
    elseif x isa AbstractMatrix
        logjac = vec(sum(log.(abs.(J)); dims = 1))  # sum along column
    end

    y = J .* x
    return (rv=y, logabsdetjac=logjac)
end
