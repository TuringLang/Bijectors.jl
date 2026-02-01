"""
    OnlyWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
vectors of length one to scalars.
"""
struct OnlyWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::OnlyWrap)(x) = w.bijector(x[])
function with_logabsdet_jacobian(w::OnlyWrap, x::AbstractVector)
    return with_logabsdet_jacobian(w.bijector, x[])
end
inverse(w::OnlyWrap) = VectWrap(inverse(w.bijector))
# Internal helper function to unify access to the wrapped bijector
get_inner_bijector(w::OnlyWrap) = w.bijector

"""
    VectWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
scalars to vectors of length one.
"""
struct VectWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::VectWrap)(x) = [w.bijector(x)]
function with_logabsdet_jacobian(w::VectWrap, x::Number)
    y, ladj = with_logabsdet_jacobian(w.bijector, x)
    return ([y], ladj)
end
inverse(w::VectWrap) = OnlyWrap(inverse(w.bijector))
get_inner_bijector(w::VectWrap) = w.bijector

# For all univariate distributions, from_vec and to_vec are simple
VectorBijectors.from_vec(::D.UnivariateDistribution) = OnlyWrap(TypedIdentity())
VectorBijectors.to_vec(::D.UnivariateDistribution) = VectWrap(TypedIdentity())

# For discrete univariate distributions, we really can't transform the 'support'
function VectorBijectors.from_linked_vec(::D.DiscreteUnivariateDistribution)
    return OnlyWrap(TypedIdentity())
end
function VectorBijectors.to_linked_vec(::D.DiscreteUnivariateDistribution)
    return VectWrap(TypedIdentity())
end

# vect_length and linked_vec_length are trivial
VectorBijectors.vec_length(::D.UnivariateDistribution) = 1
VectorBijectors.linked_vec_length(::D.UnivariateDistribution) = 1

# Optics are trivially obtainable.
VectorBijectors.optic_vec(::D.UnivariateDistribution) = [AbstractPPL.Iden()]
VectorBijectors.linked_optic_vec(::D.UnivariateDistribution) = [AbstractPPL.Iden()]
