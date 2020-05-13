struct Scale{T, N} <: Bijector{N}
    a::T
end

Base.:(==)(b1::Scale{<:Any, N}, b2::Scale{<:Any, N}) where {N} = b1.a == b2.a

function Scale(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Scale{typeof(a), D}(a)
end

up1(b::Scale{T, N}) where {N, T} = Scale{T, N + 1}(a)

(b::Scale)(x) = b.a .* x
(b::Scale{<:AbstractMatrix, 1})(x::AbstractVecOrMat) = b.a * x
(b::Scale{<:AbstractMatrix, 2})(x::AbstractMatrix) = b.a * x
(ib::Inverse{<:Scale})(y) = Scale(inv(ib.orig.a))(y)
(ib::Inverse{<:Scale{<:AbstractVector}})(y) = Scale(inv.(ib.orig.a))(y)
(ib::Inverse{<:Scale{<:AbstractMatrix, 1}})(y::AbstractVecOrMat) = ib.orig.a \ y
(ib::Inverse{<:Scale{<:AbstractMatrix, 2}})(y::AbstractMatrix) = ib.orig.a \ y

# We're going to implement custom adjoint for this
logabsdetjac(b::Scale{<:Any, N}, x) where {N} = _logabsdetjac_scale(b.a, x, Val(N))

_logabsdetjac_scale(a::Real, x::Real, ::Val{0}) = log(abs(a))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0}) = fill(log(abs(a)), length(x))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{1}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{1}) = fill(log(abs(a)) * size(x, 1), size(x, 2))
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{2}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractArray{<:AbstractMatrix}, ::Val{2}) = map(x) do x
    _logabsdetjac_scale(a, x, Val(2))
end
_logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1}) = sum(x -> log(abs(x)), a)
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1}) = fill(sum(x -> log(abs(x)), a), size(x, 2))
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{2}) = sum(x -> log(abs(x)), a)
_logabsdetjac_scale(a::AbstractVector, x::AbstractArray{<:AbstractMatrix}, ::Val{2}) = map(x) do x
    _logabsdetjac_scale(a, x, Val(2))
end
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractVector, ::Val{1}) = logabsdet(a)[1]
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix{T}, ::Val{1}) where {T} = logabsdet(a)[1] * ones(T, size(x, 2))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix, ::Val{2}) = logabsdet(a)[1]
function _logabsdetjac_scale(
    a::AbstractMatrix,
    x::AbstractArray{<:AbstractMatrix},
    ::Val{2},
)
    map(x) do x
        _logabsdetjac_scale(a, x, Val(2))
    end
end