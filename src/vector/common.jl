"""
    ScalarToScalarBijector

An abstract type for bijectors that map scalars to scalars.

Any subtype of this must implement `Bijectors.is_monotonically_increasing`
and `Bijectors.is_monotonically_decreasing`. One of them should be true and
one should be false.
"""
abstract type ScalarToScalarBijector end

"""
    TypedIdentity <: ScalarToScalarBijector

The same as `identity`.

The problem with using `identity` as a bijector is that ChangesOfVariables.jl defines
`with_logabsdet_jacobian(identity, x) = (x, zero(eltype(x)))`, which can fail if `eltype(x)`
is not a number type! Implementing this allows us to shortcircuit that definition and return
a sensible result (i.e. a Float64) even if `x` is not a numeric vector.
"""
struct TypedIdentity <: ScalarToScalarBijector end
B.is_monotonically_increasing(::TypedIdentity) = true
B.is_monotonically_decreasing(::TypedIdentity) = false
(::TypedIdentity)(@nospecialize(x)) = x
function with_logabsdet_jacobian(::TypedIdentity, x::AbstractArray{T}) where {T<:Number}
    return (x, zero(T))
end
with_logabsdet_jacobian(::TypedIdentity, x::T) where {T<:Number} = (x, zero(T))
with_logabsdet_jacobian(::TypedIdentity, x) = (x, zero(Float64))
inverse(x::TypedIdentity) = x
