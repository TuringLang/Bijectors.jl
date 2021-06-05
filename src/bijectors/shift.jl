#################
# Shift & Scale #
#################
struct Shift{T} <: Bijector
    a::T
end

Base.:(==)(b1::Shift, b2::Shift) = b1.a == b2.a

Functors.@functor Shift

transform(b::Shift, x) = b.a .+ x

inv(b::Shift) = Shift(-b.a)

# FIXME: implement custom adjoint to ensure we don't get tracking
function logabsdetjac(b::Shift, x::AbstractArray{<:Real, N}) where {N}
    return _logabsdetjac_shift(b.a, x, Val(N))
end

function logabsdetjac_batch(b::Shift, x::AbstractArray{<:Real, N}) where {N}
    return _logabsdetjac_shift(b.a, x, Val(N - 1))
end

_logabsdetjac_shift(a::Real, x::Real, ::Val{0}) = zero(eltype(x))
_logabsdetjac_shift(a::Real, x::AbstractVector{T}, ::Val{0}) where {T<:Real} = zeros(T, length(x))
_logabsdetjac_shift(a::T1, x::AbstractVector{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zero(T2)
_logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zeros(T2, size(x, 2))
_logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{2}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zero(T2)
_logabsdetjac_shift(a::T1, x::AbstractArray{<:AbstractMatrix{T2}}, ::Val{2}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zeros(T2, size(x))
