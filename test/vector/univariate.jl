const univariates = [
    Arcsine(0, 1),
    Beta(2, 2),
    BetaPrime(1, 2),
    Biweight(1, 2),
    Cauchy(-2, 1),
    Chi(1),
    Chisq(3),
    Cosine(0, 1),
    Epanechnikov(0, 1),
    Erlang(7, 0.5),
    Exponential(0.5),
    FDist(10, 1),
    Frechet(1, 1),
    Gamma(7.5, 1),
    GeneralizedExtremeValue(0, 1, 1),
    GeneralizedPareto(0, 1, 1),
    Gumbel(0, 1),
    InverseGamma(3, 0.5),
    InverseGaussian(1, 1),
    JohnsonSU(0.0, 1.0, 0.0, 1.0),
    Kolmogorov(),
    Kumaraswamy(2, 5),
    Laplace(0, 4),
    Levy(0, 1),
    Lindley(1.5),
    Logistic(2, 1),
    LogitNormal(0, 1),
    LogNormal(0, 1),
    LogUniform(1, 10),
    NoncentralBeta(2, 3, 1),
    NoncentralChisq(2, 3),
    NoncentralF(2, 3, 1),
    NoncentralT(2, 3),
    Normal(0, 1),
    NormalCanon(0, 1),
    NormalInverseGaussian(0, 0.5, 0.2, 0.1),
    Pareto(1, 1),
    PGeneralizedGaussian(0.2),
    Rayleigh(0.5),
    Rician(0.5, 1),
    Semicircle(1),
    SkewedExponentialPower(0, 1, 0.7, 0.7),
    SkewNormal(0, 1, -1),
    StudentizedRange(2, 2),
    SymTriangularDist(0, 1),
    TDist(5),
    TriangularDist(0, 1.5, 0.5),
    Triweight(1, 1),
    Uniform(0, 1),
    VonMises(0.5),
    Weibull(0.5, 1),
    truncated(Normal(); lower=0.0),
    truncated(Normal(); upper=0.0),
    truncated(Normal(); lower=0.0, upper=1.0),
    censored(Normal(); lower=0.0),
    censored(Normal(); upper=0.0),
    censored(Normal(); lower=0.0, upper=1.0),
    Bernoulli(0.5),
    BernoulliLogit(0.0),
    BetaBinomial(5, 2, 2),
    Binomial(5, 0.5),
    Categorical([0.2, 0.5, 0.3]),
    Dirac(2.5),
    DiscreteUniform(1, 10),
    DiscreteNonParametric([1, 3, 5], [0.2, 0.5, 0.3]),
    Geometric(0.3),
    Hypergeometric(20, 7, 12),
    NegativeBinomial(5, 0.5),
    Poisson(3.0),
    PoissonBinomial([0.2, 0.5, 0.3]),
    Skellam(2.0, 3.0),
    Soliton(100, 60, 0.2),
    MixtureModel([Normal(-2.0, 1.2), Normal(0.0, 1.0), Normal(3.0, 2.5)], [0.2, 0.5, 0.3]),
    MixtureModel([Normal(0, 1)], [1.0]),
    MixtureModel([Beta(2, 2), Beta(5, 1)], [0.5, 0.5]),
    Logistic() + 2,
    Logistic() - 2,
    Logistic() * 3,
    Logistic() * -3,
    Gamma(2, 3) + 2,
    Gamma(2, 3) - 2,
    Gamma(2, 3) * 3,
    Gamma(2, 3) * -3,
    Beta(2, 5) + 2,
    Beta(2, 5) - 2,
    Beta(2, 5) * 3,
    Beta(2, 5) * -3,
    truncated(Beta(2, 5); lower=0.2, upper=0.8),
    truncated(Beta(2, 5) * -4; lower=-3.0, upper=-1.0),
]

# Abstract eltype causes a loss of inference that allocates on < 1.12; skip the
# zero-allocation tests there.
const heterogeneous_mixtures = [
    MixtureModel(Union{Normal,Exponential}[Normal(0, 1), Exponential(1)], [0.4, 0.6]),
    MixtureModel(Union{Gamma,Exponential}[Gamma(2, 1), Exponential(3)], [0.5, 0.5]),
]

function _gen_testcases(::Val{:univariates})
    cases = VectorTestCase[]
    for d in univariates
        push!(cases, VectorTestCase(d; expected_zero_allocs=(from_vec, from_linked_vec)))
    end
    expected_alloc_for_hm = @static if VERSION >= v"1.12-"
        (from_vec, from_linked_vec)
    else
        ()
    end
    for d in heterogeneous_mixtures
        push!(cases, VectorTestCase(d; expected_zero_allocs=expected_alloc_for_hm))
    end
    return cases
end
