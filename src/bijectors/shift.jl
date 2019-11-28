#################
# Shift & Scale #
#################
struct Shift{T, N} <: Bijector{N}
    a::T
end

Shift(a::T; dim::Val{D} = Val(0)) where {T<:Real, D} = Shift{T, D}(a)
Shift(a::A; dim::Val{D} = Val(N)) where {T, D, N, A<:AbstractArray{T, N}} = Shift{A, N}(a)

(b::Shift)(x) = b.a + x
(b::Shift{<:Real})(x::AbstractArray) = b.a .+ x
(b::Shift{<:AbstractVector})(x::AbstractMatrix) = b.a .+ x

inv(b::Shift{T, N}) where {T, N} = Shift{T, N}(-b.a)

# FIXME: implement custom adjoint to ensure we don't get tracking
logabsdetjac(b::Shift{T, N}, x) where {T, N} = _logabsdetjac_shift(b.a, x, Val(N))

_logabsdetjac_shift(a::Real, x::Real, ::Val{0}) = zero(eltype(x))
_logabsdetjac_shift(a::Real, x::AbstractVector{T}, ::Val{0}) where {T<:Real} = zeros(T, length(x))
_logabsdetjac_shift(a::T1, x::AbstractVector{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zero(T2)
_logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zeros(T2, size(x, 2))

function _logabsdetjac_shift(a::TrackedReal, x::Real, ::Val{0})
    return Tracker.param(_logabsdetjac_shift(data(a), data(x), Val(0)))
end
function _logabsdetjac_shift(a::TrackedReal, x::AbstractVector{T}, ::Val{0}) where {T<:Real}
    return Tracker.param(_logabsdetjac_shift(data(a), data(x), Val(0)))
end
function _logabsdetjac_shift(a::T1, x::AbstractVector{T2}, ::Val{1}) where {T1<:Union{TrackedReal, TrackedVector}, T2<:Real}
    return Tracker.param(_logabsdetjac_shift(data(a), data(x), Val(1)))
end
function _logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{1}) where {T1<:Union{TrackedReal, TrackedVector}, T2<:Real}
    return Tracker.param(_logabsdetjac_shift(data(a), data(x), Val(1)))
end
