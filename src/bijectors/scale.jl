struct Scale{T, N} <: Bijector{N}
    a::T
end

function Scale(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Scale{typeof(a), D}(a)
end

(b::Scale)(x) = b.a .* x
(b::Scale{<:AbstractMatrix})(x::AbstractVecOrMat) = b.a * x
(ib::Inverse{<:Scale})(y) = Scale(inv(ib.orig.a))(y)
(ib::Inverse{<:Scale{<:AbstractVector}})(y) = Scale(inv.(ib.orig.a))(y)
function (ib::Inverse{<:Scale{<:AbstractMatrix, 1}})(y::AbstractVecOrMat)
    return ib.orig.a \ y
end

# We're going to implement custom adjoint for this
logabsdetjac(b::Scale{T, N}, x) where {T, N} = _logabsdetjac_scale(b.a, x, Val(N))

_logabsdetjac_scale(a::Real, x::Real, ::Val{0}) = log(abs(a))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0}) = fill(log(abs(a)), length(x))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{1}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{1}) = fill(log(abs(a)) * size(x, 1), size(x, 2))
_logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1}) = sum(x -> log(abs(x)), a)
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1}) = fill(sum(x -> log(abs(x)), a), size(x, 2))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractVector, ::Val{1}) = logabsdet(a)
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix{T}, ::Val{1}) where {T} = logabsdet(a) * ones(T, size(x, 2))