#################
# Shift & Scale #
#################
struct Shift{T} <: Bijector
    a::T
end

Base.:(==)(b1::Shift, b2::Shift) = b1.a == b2.a

Functors.@functor Shift

inverse(b::Shift) = Shift(-b.a)

transform(b::Shift, x) = b.a .+ x

# FIXME: implement custom adjoint to ensure we don't get tracking
function logabsdetjac(b::Shift, x::Union{Real, AbstractArray{<:Real}})
    return _logabsdetjac_shift(b.a, x)
end

_logabsdetjac_shift(a, x) = zero(eltype(x))
_logabsdetjac_shift_array_batch(a, x) = zeros(eltype(x), size(x, ndims(x)))

with_logabsdet_jacobian(b::Shift, x) = transform(b, x), logabsdetjac(b, x)
