#################
# Shift & Scale #
#################
struct Shift{T, N} <: Bijector{N}
    a::T
end

function Shift(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Shift{typeof(a), D}(a)
end

(b::Shift)(x) = b.a + x
(b::Shift{<:Real})(x::AbstractArray) = b.a .+ x
(b::Shift{<:AbstractVector})(x::AbstractMatrix) = b.a .+ x

inv(b::Shift{T, N}) where {T, N} = Shift{T, N}(-b.a)

# FIXME: implement custom adjoint to ensure we don't get tracking
logabsdetjac(b::Shift{T, N}, x) where {T, N} = _logabsdetjac_shift(b.a, x, Val(N))

_logabsdetjac_shift(a::Real, x::Real, ::Val{0}) = zero(Base.promote_typeof(a, x))
function _logabsdetjac_shift(a::Real, x::AbstractVector{<:Real}, ::Val{0})
    T = Base.promote_eltype(a, x)
    return zeros(T, length(x))
end
function _logabsdetjac_shift(a::Union{Real, AbstractVector}, x::AbstractVector{<:Real}, ::Val{1})
    T = Base.promote_eltype(a, x)
    return zero(T)
end
function _logabsdetjac_shift(a::Union{Real, AbstractVector}, x::AbstractMatrix{<:Real}, ::Val{1})
    T = Base.promote_eltype(a, x)
    return zeros(T, size(x, 2))
end