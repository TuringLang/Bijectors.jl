# Distribution lists and parameterised `test_all` wrappers used by both the main
# `test/vector/*.jl` suite (with non-Enzyme adtypes) and `test/integration/enzyme/main.jl`
# (with Enzyme adtypes). Each wrapper accepts the adtype list(s) so the same test bodies
# run against any AD backend.
#
# Requires `Bijectors`, `Bijectors.VectorBijectors`, `Distributions`, `FillArrays`,
# `LinearAlgebra`, `PDMats`, and `Test` to be loaded.

using Bijectors
using Bijectors: ordered
using Bijectors.VectorBijectors
using Distributions
using FillArrays: Fill
using LinearAlgebra
using PDMats
using Test

# ----- univariates -----

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

# Because of the abstract eltype, there's some loss of inference somewhere that causes
# allocations on < 1.12, so for these we skip the zero-allocation tests.
const heterogeneous_mixtures = [
    MixtureModel(Union{Normal,Exponential}[Normal(0, 1), Exponential(1)], [0.4, 0.6]),
    MixtureModel(Union{Gamma,Exponential}[Gamma(2, 1), Exponential(3)], [0.5, 0.5]),
]

function test_univariates_with(adtypes)
    for d in univariates
        VectorBijectors.test_all(
            d; adtypes=adtypes, expected_zero_allocs=(from_vec, from_linked_vec)
        )
    end
    for d in heterogeneous_mixtures
        expected_zero_allocs = @static if VERSION >= v"1.12-"
            (from_vec, from_linked_vec)
        else
            ()
        end
        VectorBijectors.test_all(
            d; adtypes=adtypes, expected_zero_allocs=expected_zero_allocs
        )
    end
end

# ----- multivariates -----

const multivariates = [
    Multinomial(10, [0.2, 0.5, 0.3]),
    MvNormal([0.0, 0.0], I),
    MvNormalCanon([1.0, 2.0, 3.0], [4.0 -2.0 -1.0; -2.0 5.0 -1.0; -1.0 -1.0 6.0]),
    MvTDist(5.0, zeros(2), Matrix(1.0I, 2, 2)),
    MvTDist(1.0, [1.0, -1.0, 0.5], [2.0 0.5 0.0; 0.5 3.0 0.5; 0.0 0.5 1.5]),
    MvLogNormal([0.0, 0.0], I),
    MvLogitNormal([1.0, 2.0], Diagonal([4.0, 5.0])),
    Dirichlet([2.0, 3.0, 5.0]),
]

function test_multivariates_with(adtypes)
    for d in multivariates
        expected_zero_allocs = if d isa Union{Dirichlet,MvLogitNormal,MvLogNormal}
            (to_vec, from_vec)
        else
            (to_vec, from_vec, to_linked_vec, from_linked_vec)
        end
        VectorBijectors.test_all(
            d; adtypes=adtypes, expected_zero_allocs=expected_zero_allocs
        )
    end
end

# ----- matrix distributions -----

const _matrix_ν = 5
const _matrix_M = [1 2 3; 4 5 6]
const _matrix_Σ = PDMats.PDMat([1 0.5; 0.5 1])
const _matrix_Ω = PDMats.PDMat([1 0.3 0.2; 0.3 1 0.4; 0.2 0.4 1])

const matrix_dists = [
    MatrixNormal(2, 4),
    MatrixNormal(3, 5),
    MatrixTDist(_matrix_ν, _matrix_M, _matrix_Σ, _matrix_Ω),
    Wishart(7, Matrix{Float64}(I, 2, 2)),
    Wishart(7, Matrix{Float64}(I, 4, 4)),
    InverseWishart(7, Matrix{Float64}(I, 2, 2)),
    InverseWishart(7, Matrix{Float64}(I, 4, 4)),
]

const lkj_matrix_dists = [LKJ(3, 1.0), LKJ(7, 1.0)]

function test_matrix_dists_with(adtypes; lkj_adtypes=adtypes)
    for d in matrix_dists
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end
    # LKJ runs with a possibly smaller backend list because ReverseDiff gives wrong results
    # when differentiating through VecCorrBijector
    # (https://github.com/TuringLang/Bijectors.jl/issues/434). Pass an empty
    # `lkj_adtypes` to skip LKJ entirely. Don't check `from_linked_vec(d)(randn(...))`
    # support for LKJ — numerical precision in the inverse bijector means diagonal entries
    # are not exactly 1 (https://github.com/TuringLang/Bijectors.jl/issues/435).
    if !isempty(lkj_adtypes)
        for d in lkj_matrix_dists
            VectorBijectors.test_all(
                d; adtypes=lkj_adtypes, expected_zero_allocs=(), test_in_support=false
            )
        end
    end
end

# ----- cholesky -----

# Note: can't test LKJCholesky(1, ...) because its linked vector is length-zero and
# DifferentiationInterface trips up with empty vectors.
const cholesky_dists = [
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

function test_cholesky_with(adtypes)
    for d in cholesky_dists
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end
end

# ----- reshaped -----

const reshaped_default_dists = [
    # 0-dim array output: doesn't work because
    # https://github.com/JuliaStats/Distributions.jl/issues/2025
    # reshape(Normal(), ()),
    vec(Normal()),
    reshape(Normal(), (1, 1, 1, 1, 1)),
    vec(Beta(2, 2)),
    vec(Poisson(3)),
    reshape(Poisson(3), (1, 1, 1, 1, 1)),
    reshape(MvNormal(zeros(2), I), (2, 1, 1)),
    reshape(MvNormal(zeros(4), I), (2, 2)),
    reshape(Dirichlet(ones(6)), (2, 3)),
    reshape(MatrixNormal(2, 4), 8),
    reshape(MatrixNormal(2, 5), 5, 2),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 16),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 1, 1, 4, 1, 4),
]

# reshape(Beta(2, 2), (1, 1, 1, 1, 1)) hit https://github.com/EnzymeAD/Enzyme.jl/issues/2987
# on Julia 1.10. Callers that need the special-case adtype list (e.g. Enzyme on 1.10) pass
# `beta_reshape_adtypes_pre_111`; everyone else lets it fall back to `adtypes`.
const reshaped_beta_dist = reshape(Beta(2, 2), (1, 1, 1, 1, 1))

function test_reshaped_with(adtypes; beta_reshape_adtypes_pre_111=adtypes)
    for d in reshaped_default_dists
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end
    @static if VERSION >= v"1.11-"
        VectorBijectors.test_all(
            reshaped_beta_dist; adtypes=adtypes, expected_zero_allocs=()
        )
    else
        VectorBijectors.test_all(
            reshaped_beta_dist;
            adtypes=beta_reshape_adtypes_pre_111,
            expected_zero_allocs=(),
        )
    end
end

# ----- transformed -----

const transformed_dists = [
    transformed(Normal(), exp),
    transformed(Beta(2, 3), Bijectors.Logit(0.0, 1.0)),
    transformed(Gamma(2, 1), elementwise(log)),
    transformed(product_distribution(fill(Beta(2, 2), 4)), elementwise(exp)),
    transformed(MvNormal(zeros(3), I), Bijectors.Scale(2.0)),
    transformed(Dirichlet([1.0, 2.0, 3.0])),
    transformed(MvLogNormal(zeros(2), I), elementwise(log)),
    transformed(MatrixNormal(zeros(2, 3), I(2), I(3)), elementwise(exp)),
]

function test_transformed_with(adtypes)
    for d in transformed_dists
        VectorBijectors.test_all(d; adtypes=adtypes, test_in_support=false)
    end
end

# ----- order statistics -----

const order_base_dists = [
    Normal(),
    InverseGamma(2, 3),
    InverseGamma(2, 3) * -2,
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

function test_order_with(adtypes; joint_adtypes=adtypes)
    for d in order_base_dists
        unvec_only = (from_vec, from_linked_vec)
        VectorBijectors.test_all(
            OrderStatistic(d, 10, 1); adtypes=adtypes, expected_zero_allocs=unvec_only
        )
        VectorBijectors.test_all(
            OrderStatistic(d, 10, 10); adtypes=adtypes, expected_zero_allocs=unvec_only
        )
        # JointOrderStatistics is only defined for continuous distributions.
        if d isa ContinuousUnivariateDistribution
            # In the unlinked case, the transform is identity.
            # https://github.com/TuringLang/Bijectors.jl/issues/441 explains the large atol.
            unlinked_only = (from_vec, to_vec)
            VectorBijectors.test_all(
                JointOrderStatistics(d, 4);
                expected_zero_allocs=unlinked_only,
                adtypes=joint_adtypes,
                roundtrip_atol=1e-1,
            )
            VectorBijectors.test_all(
                JointOrderStatistics(d, 10, 2:5);
                expected_zero_allocs=unlinked_only,
                adtypes=joint_adtypes,
                roundtrip_atol=1e-1,
            )
        end
    end

    VectorBijectors.test_all(
        ordered(MvNormal([0.0, 1.0, 2.0], I));
        adtypes=adtypes,
        expected_zero_allocs=(from_vec, to_vec),
    )
end

# ----- product distributions -----

const _m2 = MvNormal(zeros(2), I)
const _d2 = Dirichlet(ones(2))
const _p1t = product_distribution(Normal(), Beta(2, 2))
const _p2t = product_distribution(_m2, _d2)
const _p1a = product_distribution(fill(Beta(2, 2), 2))
const _p2a = product_distribution(fill(_d2, 2))

# Purposely chosen because the vec_length output is the same but linked_vec_length differs.
const products = [
    product_distribution(Normal()),
    product_distribution(Normal(), Normal()),
    product_distribution(Normal(), Beta(2, 2)),
    product_distribution(Beta(2, 2), Exponential()),
    product_distribution(_m2, _d2),
    product_distribution(_m2, _d2, _m2, _d2),
    product_distribution(fill(Normal(), 2)),
    product_distribution(fill(Beta(2, 2), 2)),
    product_distribution([Uniform(0, 1), Uniform(1, 2), Uniform(2, 3)]),
    product_distribution(Fill(Uniform(1, 2), 2)),
    product_distribution(fill(Normal(), 2, 2)),
    product_distribution(Fill(Uniform(1, 2), 2, 2)),
    product_distribution(fill(_m2, 2, 2)),
    product_distribution(Fill(_m2, 2, 2)),
    product_distribution(fill(_d2, 2, 2)),
    product_distribution((a=Normal(), b=Beta(2, 2))),
    product_distribution((a=Normal(), b=Dirichlet(ones(2)))),
    product_distribution((a=Normal(), b=product_distribution(fill(Beta(2, 2), 2)))),
    product_distribution(fill(_p1t, 2)),
    product_distribution(fill(_p1t, 2, 2)),
    product_distribution(_p2t, _p2t, _p2t),
    product_distribution(fill(_p2t, 2)),
    product_distribution(fill(_p2t, 2, 2)),
    product_distribution(fill(_p1a, 2)),
    product_distribution(fill(_p1a, 2, 2)),
    product_distribution(_p2a, _p2a, _p2a),
    product_distribution(fill(_p2a, 2)),
    product_distribution(fill(_p2a, 2, 2)),
]

# On Julia 1.10 (and only 1.10), @inferred to_vec(d) fails for this case, but
# @code_warntype to_vec(d) is type stable. Almost certainly a Julia bug.
const nested_product_namedtuple = [
    product_distribution((a=Normal(), b=product_distribution((c=Normal(), d=Beta(2, 2))))),
]

# Heterogeneous arrays make bijector construction type unstable. The triple-nested tuple
# products (last two) also originally lived here as "enzyme_failures" — Enzyme couldn't
# differentiate through them on `main`.
const type_unstable_products = [
    product_distribution([Normal(), Beta(2, 2), Exponential()]),
    product_distribution([Normal() Beta(2, 2); Exponential() Uniform(-1, 1)]),
    product_distribution([_m2 _d2; _m2 _d2]),
    product_distribution(_p1t, _p1t, _p1t),
    product_distribution(_p1a, _p1a, _p1a),
]

function test_products_with(adtypes)
    for d in products
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end

    for d in nested_product_namedtuple
        VectorBijectors.test_all(
            d;
            adtypes=adtypes,
            expected_zero_allocs=(),
            test_construction_type_stable=(VERSION >= v"1.11-"),
        )
    end

    for d in type_unstable_products
        VectorBijectors.test_all(
            d;
            adtypes=adtypes,
            expected_zero_allocs=(),
            test_construction_type_stable=false,
        )
    end
end
