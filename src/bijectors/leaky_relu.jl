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
    return mask * b.α * x + (1 - mask) * x
end
(b::LeakyReLU{<:Any, 0})(x::AbstractVector{<:Real}) = b.(x)

function (ib::Inverse{<:LeakyReLU, 0})(y::Real)
    mask = y < zero(y)
    return mask * (x / ib.orig.α) + (1 - mask) * x
end
(ib::Inverse{<:LeakyReLU{<:Any}, 0})(y::AbstractVector{<:Real}) = ib.(y)

function logabsdetjac(b::LeakyReLU{<:Any, 0}, x::Real)
    mask = x < zero(x)
    J = mask * b.α + (1 - mask) * one(x)
    return log.(abs.(J))
end
logabsdetjac(b::LeakyReLU{<:Real, 0}, x::AbstractVector{<:Real}) = logabsdetjac.(b, x)


# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{<:Any, 0}, x::Real)
    mask = x < zero(x)
    J = mask * b.α + (1 - mask) * one(x)
    return (rv=J * x, logabsdetjac=log(abs(J)))
end

# Batched version
function forward(b::LeakyReLU{<:Any, 0}, x::AbstractVector)
    mask = x .< zero(eltype(x))
    J = mask .* b.α .+ (1 .- mask) .* one(eltype(x))
    return (rv=J .* x, logabsdetjac=log.(abs.(J)))
end

# (N=1) Multivariate case, with univariate parameter `α`
function (b::LeakyReLU{<:Any, 1})(x::AbstractVecOrMat)
    mask = x .< zero(eltype(x))
    return mask .* b.α .* x .+ (1 .- mask) .* x
end

function (ib::Inverse{<:LeakyReLU, 1})(y::AbstractVecOrMat)
    mask = x .< zero(eltype(y))
    return mask .* (y ./ ib.orig.α) .+ (1 .- mask) .* y
end

function logabsdetjac(b::LeakyReLU{<:Any, 1}, x::AbstractVecOrMat)
    # Is really diagonal of jacobian
    mask = x .< zero(eltype(x))
    J = mask .* b.α .+ (1 .- mask) .* one(eltype(x))

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
    mask = x .< zero(eltype(x))
    J = mask .* b.α .+ (1 .- mask) .* one(eltype(x))

    if x isa AbstractVector
        logjac = sum(log.(abs.(J)))
    elseif x isa AbstractMatrix
        logjac = vec(sum(log.(abs.(J)); dims = 1))  # sum along column
    end

    y = J .* x
    return (rv=y, logabsdetjac=logjac)
end
