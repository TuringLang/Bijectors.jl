module VBUnivariateTests

using Distributions
using Test
using Bijectors.VectorBijectors
using ..VectorTestUtils

dists = [
    Arcsine(0, 1), # trunc
    Beta(2, 2), # trunc
    BetaPrime(1, 2), # pos
    Biweight(1, 2),
    Cauchy(-2, 1), # iden
    # sampling from Chernoff errors randomly:
    # https://github.com/JuliaStats/Distributions.jl/issues/1999
    # https://github.com/JuliaStats/Distributions.jl/pull/2000
    # Chernoff(), # iden
    Chi(1), # pos
    Chisq(3), # pos
    Cosine(0, 1), # trunc
    Epanechnikov(0, 1),
    Erlang(7, 0.5), # pos
    Exponential(0.5), # pos
    FDist(10, 1), # pos
    Frechet(1, 1), # trunc
    Gamma(7.5, 1), # pos
    GeneralizedExtremeValue(0, 1, 1), # trunc
    GeneralizedPareto(0, 1, 1), # trunc
    Gumbel(0, 1), # iden
    InverseGamma(3, 0.5), # pos
    InverseGaussian(1, 1), # pos
    JohnsonSU(0.0, 1.0, 0.0, 1.0), # iden
    Kolmogorov(), # pos
    # KSDist(5), # can't rand
    # KSOneSided(5), # can't rand
    Kumaraswamy(2, 5), # trunc
    Laplace(0, 4), # iden
    Levy(0, 1), # trunc
    Lindley(1.5), # pos
    Logistic(2, 1), # iden
    LogitNormal(0, 1), # trunc
    LogNormal(0, 1), # pos
    LogUniform(1, 10), # trunc
    NoncentralBeta(2, 3, 1), # trunc
    NoncentralChisq(2, 3), # pos
    NoncentralF(2, 3, 1), # pos
    NoncentralT(2, 3), # iden
    Normal(0, 1), # iden
    NormalCanon(0, 1), # iden
    NormalInverseGaussian(0, 0.5, 0.2, 0.1), # iden
    Pareto(1, 1), # trunc
    PGeneralizedGaussian(0.2), # iden
    Rayleigh(0.5), # pos
    Rician(0.5, 1), # pos
    Semicircle(1), # trunc
    SkewedExponentialPower(0, 1, 0.7, 0.7), # iden
    SkewNormal(0, 1, -1), # iden
    StudentizedRange(2, 2), # pos
    SymTriangularDist(0, 1), # trunc
    TDist(5), # iden
    TriangularDist(0, 1.5, 0.5), # trunc
    Triweight(1, 1), # trunc
    Uniform(0, 1), # trunc
    VonMises(0.5), # trunc
    Weibull(0.5, 1), # pos
    truncated(Normal(); lower=0.0), # trunc
    truncated(Normal(); upper=0.0), # trunc
    truncated(Normal(); lower=0.0, upper=1.0), # trunc
    censored(Normal(); lower=0.0), # trunc
    censored(Normal(); upper=0.0), # trunc
    censored(Normal(); lower=0.0, upper=1.0), # trunc
]

@testset "Univariates" begin
    for d in dists
        VectorTestUtils.test_all(d; expected_zero_allocs=(from_vec, from_linked_vec))
    end
end

end # module VBUnivariateTests
