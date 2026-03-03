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

# For all univariate distributions, from_vec and to_vec are simple
VectorBijectors.from_vec(::D.UnivariateDistribution) = OnlyWrap(TypedIdentity())
VectorBijectors.to_vec(::D.UnivariateDistribution) = VectWrap(TypedIdentity())

# For the linked transformations, we start by defining the scalar to scalar bijector that
# maps from the support of the distribution to the real line, and then wrap it in `OnlyWrap`
# and `VectWrap` to get the vector to scalar and scalar to vector bijectors, respectively.
function scalar_to_scalar_bijector end
function VectorBijectors.from_linked_vec(d::D.UnivariateDistribution)
    return OnlyWrap(inverse(scalar_to_scalar_bijector(d)))
end
function VectorBijectors.to_linked_vec(d::D.UnivariateDistribution)
    return VectWrap(scalar_to_scalar_bijector(d))
end

# vect_length and linked_vec_length are trivial
VectorBijectors.vec_length(::D.UnivariateDistribution) = 1
VectorBijectors.linked_vec_length(::D.UnivariateDistribution) = 1

# Optics are trivially obtainable.
VectorBijectors.optic_vec(::D.UnivariateDistribution) = [AbstractPPL.Iden()]
VectorBijectors.linked_optic_vec(::D.UnivariateDistribution) = [AbstractPPL.Iden()]
