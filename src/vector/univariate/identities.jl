# These continuous distributions have support over the entire real line.
const IDENTITY_UNIVARIATES = Union{
    D.Cauchy,
    D.Chernoff,
    D.Gumbel,
    D.JohnsonSU,
    D.Laplace,
    D.Logistic,
    D.NoncentralT,
    D.Normal,
    D.NormalCanon,
    D.NormalInverseGaussian,
    D.PGeneralizedGaussian,
    D.SkewedExponentialPower,
    D.SkewNormal,
    D.TDist,
    # For discrete distributions, we can't really do any 'transformation'
    D.DiscreteUnivariateDistribution,
}

VectorBijectors.scalar_to_scalar_bijector(::IDENTITY_UNIVARIATES) = TypedIdentity()

# Furthermore, scaling and shifting doesn't affect the support of these distributions
function VectorBijectors.scalar_to_scalar_bijector(
    ::D.AffineDistribution{<:Any,<:Any,<:IDENTITY_UNIVARIATES}
)
    return TypedIdentity()
end
