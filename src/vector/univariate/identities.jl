# These continuous distributions have support over the entire real line.
for dist_type in [
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
]
    @eval begin
        VectorBijectors.from_linked_vec(::$dist_type) = Only()
        VectorBijectors.to_linked_vec(::$dist_type) = Vect()
    end
end
