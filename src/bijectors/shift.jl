#################
# Shift & Scale #
#################
struct Shift{T, N} <: Bijector{N}
    a::T
end

Base.:(==)(b1::Shift{<:Any, N}, b2::Shift{<:Any, N}) where {N} = b1.a == b2.a

function Shift(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Shift{typeof(a), D}(a)
end

# field is a numerical parameter
function Functors.functor(::Type{<:Shift{<:Any,N}}, x) where N
    function reconstruct_shift(xs)
        return Shift{typeof(xs.a),N}(xs.a)
    end
    return (a = x.a,), reconstruct_shift
end

up1(b::Shift{T, N}) where {T, N} = Shift{T, N + 1}(b.a)

(b::Shift)(x) = b.a .+ x
(b::Shift{<:Any, 2})(x::AbstractArray{<:AbstractMatrix}) = map(b, x)

InverseFunctions.inverse(b::Shift{T, N}) where {T, N} = Shift{T, N}(-b.a)

# FIXME: implement custom adjoint to ensure we don't get tracking
logabsdetjac(b::Shift{T, N}, x) where {T, N} = _logabsdetjac_shift(b.a, x, Val(N))

_logabsdetjac_shift(a::Real, x::Real, ::Val{0}) = zero(eltype(x))
_logabsdetjac_shift(a::Real, x::AbstractVector{T}, ::Val{0}) where {T<:Real} = zeros(T, length(x))
_logabsdetjac_shift(a::T1, x::AbstractVector{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zero(T2)
_logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{1}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zeros(T2, size(x, 2))
_logabsdetjac_shift(a::T1, x::AbstractMatrix{T2}, ::Val{2}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zero(T2)
_logabsdetjac_shift(a::T1, x::AbstractArray{<:AbstractMatrix{T2}}, ::Val{2}) where {T1<:Union{Real, AbstractVector}, T2<:Real} = zeros(T2, size(x))
