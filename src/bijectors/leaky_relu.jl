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

# (N=0) Univariate case
function (b::LeakyReLU{T1, 0})(x::T2) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    return T(ifelse(x < zero(T2), b.α * x, x))
end
(b::LeakyReLU{<:Real, 0})(x::AbstractVector{<:Real}) = b.(x)

function (ib::Inversed{<:LeakyReLU{T1}, 0})(y::T2) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    return T(ifelse(y < zero(T2), inv(ib.orig.α) * y, y))
end
(ib::Inversed{<:LeakyReLU{<:Real}, 0})(y::AbstractVector{<:Real}) = ib.(y)

function logabsdetjac(b::LeakyReLU{T1, 0}, x::T2) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    J⁻¹ = T(ifelse(x < 0, b.α, one(T2)))

    return log.(abs.(J⁻¹))
end
logabsdetjac(b::LeakyReLU{<:Real, 0}, x::AbstractVector{<:Real}) = logabsdetjac.(b, x)


# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{T1, 0}, x::T2) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    J = T(ifelse(x < 0, b.α, one(T2))) # <= is really diagonal of jacobian
    return (rv=J * x, logabsdetjac=log(abs(J)))
end

# Batched version
function forward(b::LeakyReLU{T1, 0}, x::AbstractVector{T2}) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)
    J = @. T(ifelse(x < 0, b.α, one(T2))) # <= is really diagonal of jacobian
    return (rv=J .* x, logabsdetjac=log.(abs.(J)))
end

# (N=1) Multivariate case, with univariate parameter `α`
function (b::LeakyReLU{T1, 1})(x::AbstractVecOrMat{T2}) where {T1<:Real, T2}
    # Note that this will do the correct thing even for `Tracker.jl` in the sense
    # that the resulting array will be `TrackedArray` rather than `Array{<:TrackedReal}`.
    T = promote_type(T1, T2)
    return @. T(ifelse(x < zero(T2), b.α * x, x))
end

function (ib::Inversed{<:LeakyReLU{T1}, 1})(y::AbstractVecOrMat{T2}) where {T1<:Real, T2<:Real}
    # Note that this will do the correct thing even for `Tracker.jl` in the sense
    # that the resulting array will be `TrackedArray` rather than `Array{<:TrackedReal}`.
    T = promote_type(T1, T2)
    return @. T(ifelse(y < zero(T2), inv(ib.orig.α) * y, y))
end

function logabsdetjac(b::LeakyReLU{T1, 1}, x::AbstractVecOrMat{T2}) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)

    # Is really diagonal of jacobian
    J⁻¹ = @. T(ifelse(x < 0, b.α, one(T2)))

    if x isa AbstractVector
        return sum(log.(abs.(J⁻¹)))
    elseif x isa AbstractMatrix
        return vec(sum(log.(abs.(J⁻¹)); dims = 1))  # sum along column
    end
end

# We implement `forward` by hand since we can re-use the computation of
# the Jacobian of the transformation. This will lead to faster sampling
# when using `rand` on a `TransformedDistribution` making use of `LeakyReLU`.
function forward(b::LeakyReLU{T1, 1}, x::AbstractVecOrMat{T2}) where {T1<:Real, T2<:Real}
    # Note that this will do the correct thing even for `Tracker.jl` in the sense
    # that the resulting array will be `TrackedArray` rather than `Array{<:TrackedReal}`.
    T = promote_type(T1, T2)

    J = @. T(ifelse(x < 0, b.α, one(T2))) # <= is really diagonal of jacobian

    if x isa AbstractVector
        logjac = sum(log.(abs.(J)))
    elseif x isa AbstractMatrix
        logjac = vec(sum(log.(abs.(J)); dims = 1))  # sum along column
    end

    y = J .* x
    return (rv=y, logabsdetjac=logjac)
end
