# Shared test resources used by the main suite (`test/ad/*.jl`, `test/vector/*.jl`) and
# the Enzyme integration test (`test/integration/enzyme/main.jl`).
#
# Tests are exposed as `TestCase` values keyed by tag: `generate_testcases(Val{:tag})`
# returns the list of cases for a tag. Each entry-point file loops over its own adtype
# list(s) and runs the same cases, so every test body lives in one place and gets exercised
# against every AD backend that opts in.
#
# Tags fall into two groups:
#
#   `ADTestCase` (bijector-specific function/argument pairs, run via `test_ad`):
#     :veccorrbijector, :veccholeskybijector, :planarlayer, :pdvecbijector, :stackedbijector
#
#   `VectorTestCase` (distributions exercised via `VectorBijectors.test_all`):
#     :univariates, :multivariates, :matrix_dists, :lkj_matrix_dists, :cholesky_dists,
#     :reshaped_dists, :reshaped_beta_special, :transformed_dists, :order_orderstatistic,
#     :order_joint, :order_ordered, :products, :nested_product_namedtuple,
#     :type_unstable_products

using Bijectors
using Bijectors: ordered
import Bijectors.VectorBijectors
using Bijectors.VectorBijectors: from_linked_vec, from_vec, to_linked_vec, to_vec
using DifferentiationInterface
using DifferentiationInterface: gradient
using Distributions
using FillArrays: Fill
using FiniteDifferences: central_fdm
using LinearAlgebra
using PDMats
using Test

# Compact, readable name for a distribution. `Bijectors.VectorBijectors._name` already
# handles wrappers (Truncated, Censored, ReshapedDistribution, OrderStatistic, etc.); fall
# back to a stripped `nameof` for everything else so `VectorTestCase.name` doesn't carry
# the 1000-character `string(d)` form for nested products.
_case_name(d) = string(VectorBijectors._name(d))

# Baseline adtype list shared by every main-suite caller. Backends with separate
# integration suites (currently: Enzyme, in test/integration/enzyme) are not in this list.
const adtypes = [
    AutoReverseDiff(),
    AutoReverseDiff(; compile=true),
    AutoMooncake(),
    AutoMooncakeForward(),
]

# ===== Test case types =====

struct ADTestCase
    name::String
    func::Function
    arg::Any
    broken::Bool
end

ADTestCase(name::String, func, arg; broken::Bool=false) = ADTestCase(name, func, arg, broken)

struct VectorTestCase{D<:Distributions.Distribution}
    name::String
    dist::D
    test_kwargs::NamedTuple
end

function VectorTestCase(name::String, dist::Distributions.Distribution; kwargs...)
    return VectorTestCase(name, dist, NamedTuple(kwargs))
end
VectorTestCase(dist::Distributions.Distribution; kwargs...) =
    VectorTestCase(_case_name(dist), dist; kwargs...)

# `generate_testcases(Val{:tag})` returns the list of cases for that tag. The method table
# is populated below per tag.
function generate_testcases end

# ===== AD helpers and runners =====

const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))

function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
    @info "testing AD for function $f with $backend"
    ref_gradient = gradient(f, REF_BACKEND, x)
    ad_gradient = gradient(f, backend, x)
    @test isapprox(ad_gradient, ref_gradient; rtol=rtol, atol=atol)
end

function run_ad_case(c::ADTestCase, adtype; rtol=1e-6, atol=1e-6)
    @testset "$(c.name)" begin
        if c.broken
            @test_broken false
        else
            test_ad(c.func, adtype, c.arg; rtol=rtol, atol=atol)
        end
    end
end

function run_vector_case(c::VectorTestCase, adtypes)
    return VectorBijectors.test_all(c.dist; adtypes=adtypes, c.test_kwargs...)
end

# ===== Bijector-specific AD test cases =====

function generate_testcases(::Val{:veccorrbijector})
    cases = ADTestCase[]
    for d in (1, 2, 4)
        dist = LKJ(d, 2.0)
        b = bijector(dist)
        binv = inverse(b)
        x = rand(dist)
        y = b(x)
        roundtrip = let b = b, binv = binv
            y_ -> sum(transform(b, binv(y_)))
        end
        inverse_only = let binv = binv
            y_ -> sum(transform(binv, y_))
        end
        push!(cases, ADTestCase("VecCorrBijector d=$d roundtrip", roundtrip, y))
        push!(cases, ADTestCase("VecCorrBijector d=$d inverse", inverse_only, y))
    end
    return cases
end

function generate_testcases(::Val{:veccholeskybijector})
    cases = ADTestCase[]
    for d in (1, 2, 4), uplo in ('U', 'L')
        dist = LKJCholesky(d, 2.0, uplo)
        b = bijector(dist)
        binv = inverse(b)
        x = rand(dist)
        y = b(x)
        cholesky_to_triangular =
            uplo == 'U' ? Bijectors.cholesky_upper : Bijectors.cholesky_lower
        roundtrip = let b = b, binv = binv
            y_ -> sum(transform(b, binv(y_)))
        end
        # `cholesky_upper`/`cholesky_lower` is needed because `sum` on a
        # LinearAlgebra.Cholesky doesn't return a scalar
        inverse_only = let binv = binv, f = cholesky_to_triangular
            y_ -> sum(f(transform(binv, y_)))
        end
        push!(cases, ADTestCase("VecCholeskyBijector d=$d uplo=$uplo roundtrip", roundtrip, y))
        push!(
            cases, ADTestCase("VecCholeskyBijector d=$d uplo=$uplo inverse", inverse_only, y)
        )
    end
    return cases
end

function generate_testcases(::Val{:planarlayer})
    # logpdf of a flow with a planar layer and two-dimensional inputs
    f = function (θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = θ[6:7]
        return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
    end
    g = function (θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), layer)
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end
    # logpdf of a flow with the inverse of a planar layer and two-dimensional inputs
    finv = function (θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), inverse(layer))
        x = θ[6:7]
        return logpdf(flow.dist, x) - logabsdetjac(flow.transform, x)
    end
    ginv = function (θ)
        layer = PlanarLayer(θ[1:2], θ[3:4], θ[5:5])
        flow = transformed(MvNormal(zeros(2), I), inverse(layer))
        x = reshape(θ[6:end], 2, :)
        return sum(logpdf(flow.dist, x) - logabsdetjac(flow.transform, x))
    end
    return [
        ADTestCase("PlanarLayer logpdf vector input", f, randn(7)),
        ADTestCase("PlanarLayer logpdf matrix input", g, randn(11)),
        ADTestCase("PlanarLayer inverse logpdf vector input", finv, randn(7)),
        ADTestCase("PlanarLayer inverse logpdf matrix input", ginv, randn(11)),
    ]
end

function generate_testcases(::Val{:pdvecbijector})
    _topd(x) = x * x' + I
    d = 4
    b = Bijectors.PDVecBijector()
    binv = inverse(b)
    z = randn(d, d)
    x = _topd(z)
    y = b(x)
    forward_only = let b = b, _topd = _topd, d = d
        x_ -> sum(transform(b, _topd(reshape(x_, d, d))))
    end
    inverse_only = let binv = binv
        y_ -> sum(transform(binv, y_))
    end
    inverse_chol_lower = let binv = binv
        y_ -> sum(Bijectors.cholesky_lower(transform(binv, y_)))
    end
    inverse_chol_upper = let binv = binv
        y_ -> sum(Bijectors.cholesky_upper(transform(binv, y_)))
    end
    return [
        ADTestCase("PDVecBijector forward", forward_only, vec(z)),
        ADTestCase("PDVecBijector inverse", inverse_only, y),
        ADTestCase("PDVecBijector inverse + cholesky_lower", inverse_chol_lower, y),
        ADTestCase("PDVecBijector inverse + cholesky_upper", inverse_chol_upper, y),
    ]
end

function generate_testcases(::Val{:stackedbijector})
    dist1 = Dirichlet(4, 1.0)
    b1 = bijector(dist1)
    dist2 = LogNormal(0.0, 1.0)
    b2 = bijector(dist2)
    x1 = rand(dist1)
    x2 = rand(dist2)
    y1 = b1(x1)
    y2 = b2(x2)

    b_tuple = Stacked((b1, b2), (1:4, 5:5))
    binv_tuple = inverse(b_tuple)
    b_vec = Stacked([b1, b2], [1:4, 5:5])
    binv_vec = inverse(b_vec)

    y = vcat(y1, [y2])

    return [
        ADTestCase(
            "StackedBijector tuple roundtrip",
            let b = b_tuple, binv = binv_tuple
                y_ -> sum(transform(b, binv(y_)))
            end,
            y,
        ),
        ADTestCase(
            "StackedBijector tuple inverse",
            let binv = binv_tuple
                y_ -> sum(transform(binv, y_))
            end,
            y,
        ),
        ADTestCase(
            "StackedBijector vector roundtrip",
            # Note: matches `main` — uses the tuple-form `binv` inside, not `binv_vec`.
            let bvec = b_vec, binv = binv_tuple
                y_ -> sum(transform(bvec, binv(y_)))
            end,
            y,
        ),
        ADTestCase(
            "StackedBijector vector inverse",
            let bvec_inv = binv_vec
                y_ -> sum(transform(bvec_inv, y_))
            end,
            y,
        ),
    ]
end

# ===== Vector distribution test cases =====

# --- univariates ---

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

function generate_testcases(::Val{:univariates})
    cases = VectorTestCase[]
    for d in univariates
        push!(
            cases, VectorTestCase(d; expected_zero_allocs=(from_vec, from_linked_vec))
        )
    end
    expected_alloc_for_hm = @static if VERSION >= v"1.12-"
        (from_vec, from_linked_vec)
    else
        ()
    end
    for d in heterogeneous_mixtures
        push!(
            cases, VectorTestCase(d; expected_zero_allocs=expected_alloc_for_hm)
        )
    end
    return cases
end

# --- multivariates ---

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

function generate_testcases(::Val{:multivariates})
    cases = VectorTestCase[]
    for d in multivariates
        expected_zero_allocs = if d isa Union{Dirichlet,MvLogitNormal,MvLogNormal}
            (to_vec, from_vec)
        else
            (to_vec, from_vec, to_linked_vec, from_linked_vec)
        end
        push!(
            cases, VectorTestCase(d; expected_zero_allocs=expected_zero_allocs)
        )
    end
    return cases
end

# --- matrix distributions ---

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

function generate_testcases(::Val{:matrix_dists})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in matrix_dists]
end

# LKJ runs with a smaller backend list because ReverseDiff gives wrong results when
# differentiating through VecCorrBijector
# (https://github.com/TuringLang/Bijectors.jl/issues/434). Don't check
# `from_linked_vec(d)(randn(...))` support — numerical precision in the inverse bijector
# means diagonal entries are not exactly 1
# (https://github.com/TuringLang/Bijectors.jl/issues/435).
function generate_testcases(::Val{:lkj_matrix_dists})
    return [
        VectorTestCase(d; expected_zero_allocs=(), test_in_support=false) for
        d in lkj_matrix_dists
    ]
end

# --- cholesky ---

# Can't test LKJCholesky(1, ...) because its linked vector is length-zero and
# DifferentiationInterface trips up with empty vectors.
const cholesky_dists = [
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

function generate_testcases(::Val{:cholesky_dists})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in cholesky_dists]
end

# --- reshaped ---

const reshaped_default_dists = [
    # 0-dim array output is blocked by
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

function generate_testcases(::Val{:reshaped_dists})
    return [
        VectorTestCase(d; expected_zero_allocs=()) for
        d in reshaped_default_dists
    ]
end

# `reshape(Beta(2, 2), (1, 1, 1, 1, 1))` hit
# https://github.com/EnzymeAD/Enzyme.jl/issues/2987 on Julia 1.10 — Enzyme Reverse fails
# there, so callers may need a smaller adtype list for this one case.
const reshaped_beta_dist = reshape(Beta(2, 2), (1, 1, 1, 1, 1))

function generate_testcases(::Val{:reshaped_beta_special})
    return [VectorTestCase(reshaped_beta_dist; expected_zero_allocs=())]
end

# --- transformed ---

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

function generate_testcases(::Val{:transformed_dists})
    return [
        VectorTestCase(d; test_in_support=false) for d in transformed_dists
    ]
end

# --- order statistics ---

const order_base_dists = [
    Normal(),
    InverseGamma(2, 3),
    InverseGamma(2, 3) * -2,
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

function generate_testcases(::Val{:order_orderstatistic})
    cases = VectorTestCase[]
    for d in order_base_dists
        unvec_only = (from_vec, from_linked_vec)
        push!(
            cases,
            VectorTestCase(
                "OrderStatistic($(_case_name(d)), 10, 1)",
                OrderStatistic(d, 10, 1);
                expected_zero_allocs=unvec_only,
            ),
        )
        push!(
            cases,
            VectorTestCase(
                "OrderStatistic($(_case_name(d)), 10, 10)",
                OrderStatistic(d, 10, 10);
                expected_zero_allocs=unvec_only,
            ),
        )
    end
    return cases
end

# JointOrderStatistics is only defined for continuous distributions. In the unlinked case
# the transform is identity. https://github.com/TuringLang/Bijectors.jl/issues/441 explains
# the unusually large `roundtrip_atol`.
function generate_testcases(::Val{:order_joint})
    cases = VectorTestCase[]
    unlinked_only = (from_vec, to_vec)
    for d in order_base_dists
        d isa ContinuousUnivariateDistribution || continue
        push!(
            cases,
            VectorTestCase(
                "JointOrderStatistics($(_case_name(d)), 4)",
                JointOrderStatistics(d, 4);
                expected_zero_allocs=unlinked_only,
                roundtrip_atol=1e-1,
            ),
        )
        push!(
            cases,
            VectorTestCase(
                "JointOrderStatistics($(_case_name(d)), 10, 2:5)",
                JointOrderStatistics(d, 10, 2:5);
                expected_zero_allocs=unlinked_only,
                roundtrip_atol=1e-1,
            ),
        )
    end
    return cases
end

function generate_testcases(::Val{:order_ordered})
    d = ordered(MvNormal([0.0, 1.0, 2.0], I))
    return [VectorTestCase("ordered(MvNormal)", d; expected_zero_allocs=(from_vec, to_vec))]
end

# --- product distributions ---

const _m2 = MvNormal(zeros(2), I)
const _d2 = Dirichlet(ones(2))
const _p1t = product_distribution(Normal(), Beta(2, 2))
const _p2t = product_distribution(_m2, _d2)
const _p1a = product_distribution(fill(Beta(2, 2), 2))
const _p2a = product_distribution(fill(_d2, 2))

# Purposely chosen so that `vec_length` agrees but `linked_vec_length` differs.
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

# On Julia 1.10 (and only 1.10), `@inferred to_vec(d)` fails for this case even though
# `@code_warntype to_vec(d)` is type stable. Almost certainly a Julia bug.
const nested_product_namedtuple = [
    product_distribution((a=Normal(), b=product_distribution((c=Normal(), d=Beta(2, 2))))),
]

# Heterogeneous arrays make bijector construction type unstable. The triple-nested tuple
# products (last two) were the `enzyme_failures` on `main` — Enzyme can't differentiate
# through them; callers that test Enzyme should filter them out.
const type_unstable_products = [
    product_distribution([Normal(), Beta(2, 2), Exponential()]),
    product_distribution([Normal() Beta(2, 2); Exponential() Uniform(-1, 1)]),
    product_distribution([_m2 _d2; _m2 _d2]),
    product_distribution(_p1t, _p1t, _p1t),
    product_distribution(_p1a, _p1a, _p1a),
]

function generate_testcases(::Val{:products})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in products]
end

function generate_testcases(::Val{:nested_product_namedtuple})
    return [
        VectorTestCase(
            d;
            expected_zero_allocs=(),
            test_construction_type_stable=(VERSION >= v"1.11-"),
        ) for d in nested_product_namedtuple
    ]
end

function generate_testcases(::Val{:type_unstable_products})
    return [
        VectorTestCase(
            d;
            expected_zero_allocs=(),
            test_construction_type_stable=false,
        ) for d in type_unstable_products
    ]
end
