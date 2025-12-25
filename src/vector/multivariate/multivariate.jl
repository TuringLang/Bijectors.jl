"""
    TypedIdentity

The same as `identity`.

The problem with using `identity` as a bijector is that ChangesOfVariables.jl defines
`with_logabsdet_jacobian(identity, x) = (x, zero(eltype(x)))`, which can fail if `eltype(x)`
is not a number type! Implementing this allows us to shortcircuit that definition and return
a sensible result (i.e. a Float64) even if `x` is not a numeric vector.
"""
struct TypedIdentity end
(::TypedIdentity)(@nospecialize(x)) = x
function with_logabsdet_jacobian(::TypedIdentity, x::AbstractArray{T}) where {T<:Number}
    return (x, zero(T))
end
with_logabsdet_jacobian(::TypedIdentity, x::T) where {T<:Number} = (x, zero(T))
with_logabsdet_jacobian(::TypedIdentity, x) = (x, zero(Float64))
inverse(x::TypedIdentity) = x

# For all multivariate distributions, from_vec and to_vec are just the identity function.
from_vec(::D.MultivariateDistribution) = TypedIdentity()
to_vec(::D.MultivariateDistribution) = TypedIdentity()
# which makes vec_length trivial
vec_length(d::D.MultivariateDistribution) = length(d)

# For discrete multivariate distributions, we really can't transform the 'support'.
from_linked_vec(::D.DiscreteMultivariateDistribution) = TypedIdentity()
to_linked_vec(::D.DiscreteMultivariateDistribution) = TypedIdentity()
linked_vec_length(d::D.DiscreteMultivariateDistribution) = length(d)
