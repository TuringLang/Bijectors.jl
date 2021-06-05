struct Scale{T} <: Bijector
    a::T
end

Base.:(==)(b1::Scale, b2::Scale) = b1.a == b2.a

Functors.@functor Scale

transform(b::Scale, x) = b.a .* x
transform(b::Scale{<:AbstractMatrix}, x::AbstractVecOrMat) = b.a * x
transform(ib::Inverse{<:Scale}, y) = transform(Scale(inv(ib.orig.a)), y)
transform(ib::Inverse{<:Scale{<:AbstractVector}}, y) = transform(Scale(inv.(ib.orig.a)), y)
transform(ib::Inverse{<:Scale{<:AbstractMatrix}}, y::AbstractVecOrMat) = ib.orig.a \ y

# We're going to implement custom adjoint for this
logabsdetjac(b::Scale, x::Real) = _logabsdetjac_scale(b.a, x, Val(0))
function logabsdetjac(b::Scale, x::AbstractArray{<:Real, N}) where {N}
    return _logabsdetjac_scale(b.a, x, Val(N))
end

function logabsdetjac_batch(b::Scale, x::ArrayBatch{N}) where {N}
    return Batch(_logabsdetjac_scale(b.a, value(x), Val(N - 1)))
end

# Scalar: single input.
_logabsdetjac_scale(a::Real, x::Real, ::Val{0}) = log(abs(a))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{1}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{2}) = log(abs(a)) * length(x)

# Scalar: batch.
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0}) = fill(log(abs(a)), length(x))
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{1}) = fill(log(abs(a)) * size(x, 1), size(x, 2))

# Vector: single input.
_logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1}) = sum(log ∘ abs, a)
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{2}) = sum(log ∘ abs, a)

# Vector: batch.
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1}) = fill(sum(log ∘ abs, a), size(x, 2))

# Matrix: single input.
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractVector, ::Val{1}) = logabsdet(a)[1]
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix, ::Val{2}) = logabsdet(a)[1]

# Matrix: batch.
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix{T}, ::Val{1}) where {T} = logabsdet(a)[1] * ones(T, size(x, 2))
