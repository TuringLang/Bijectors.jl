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
}
VectorBijectors.from_linked_vec(::IDENTITY_UNIVARIATES) = OnlyWrap(TypedIdentity())
VectorBijectors.to_linked_vec(::IDENTITY_UNIVARIATES) = VectWrap(TypedIdentity())

# Scaling and shifting doesn't affect the support of these distributions
function VectorBijectors.from_linked_vec(
    ::D.AffineDistribution{<:Any,<:Any,<:IDENTITY_UNIVARIATES}
)
    return OnlyWrap(TypedIdentity())
end
function VectorBijectors.to_linked_vec(
    ::D.AffineDistribution{<:Any,<:Any,<:IDENTITY_UNIVARIATES}
)
    return VectWrap(TypedIdentity())
end
