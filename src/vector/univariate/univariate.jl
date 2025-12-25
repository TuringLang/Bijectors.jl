"""
    Only

Callable struct, defined such that `(::Only)(x) = x[]`.

!!! warning
    This does not check whether the input has exactly one element.
"""
struct Only end
(::Only)(x) = x[]
with_logabsdet_jacobian(::Only, x::AbstractVector{T}) where {T<:Number} = (x[], zero(T))
with_logabsdet_jacobian(::Only, x::AbstractVector) = (x[], zero(Float64))
inverse(::Only) = Vect()

"""
   Vect

Callable struct, defined such that `(::Vect)(x) = [x]`.

!!! warning
    This does not check whether the input is a scalar.
"""
struct Vect end
(::Vect)(x) = [x]
with_logabsdet_jacobian(::Vect, x::Number) = ([x], zero(x))
with_logabsdet_jacobian(::Vect, x) = ([x], zero(Float64))
inverse(::Vect) = Only()

# For all univariate distributions, from_vec and to_vec are simple
VectorBijectors.from_vec(::D.UnivariateDistribution) = Only()
VectorBijectors.to_vec(::D.UnivariateDistribution) = Vect()

# For discrete univariate distributions, we really can't transform the 'support'
VectorBijectors.from_linked_vec(::D.DiscreteUnivariateDistribution) = Only()
VectorBijectors.to_linked_vec(::D.DiscreteUnivariateDistribution) = Vect()

# vect_length and linked_vec_length are trivial
VectorBijectors.vec_length(::D.UnivariateDistribution) = 1
VectorBijectors.linked_vec_length(::D.UnivariateDistribution) = 1
