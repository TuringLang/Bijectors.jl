struct Scale{T, N} <: Bijector{N}
    a::T
end

function Scale(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Scale{typeof(a), D}(a)
end

(b::Scale)(x) = b.a .* x
(b::Scale{<:Real})(x::AbstractArray) = b.a .* x
(b::Scale{<:AbstractMatrix})(x::AbstractArray) = b.a * x
(b::Scale{<:AbstractVector{<:Real}, 2})(x::AbstractMatrix{<:Real}) = b.a .* x

inv(b::Scale{T, D}) where {T, D} = Scale(inv(b.a); dim = Val(D))
inv(b::Scale{<:AbstractVector, D}) where {D} = Scale(inv.(b.a); dim = Val(D))

# We're going to implement custom adjoint for this
logabsdetjac(b::Scale{T, N}, x) where {T, N} = _logabsdetjac_scale(b.a, x, Val(N))

_logabsdetjac_scale(a::Real, x::Real, ::Val{0}) = log(abs(a))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0}) = fill(log(abs(a)), length(x))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{1}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{1}) = fill(log(abs(a)) * size(x, 1), size(x, 2))
_logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1}) = sum(log.(abs.(a)))
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1}) = fill(sum(log.(abs.(a))), size(x, 2))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractVector, ::Val{1}) = log(abs(det(a)))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix{T}, ::Val{1}) where {T} = log(abs(det(a))) * ones(T, size(x, 2))