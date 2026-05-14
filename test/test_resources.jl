# Shared test resources used by `test/vector_bijectors.jl` (main suite) and the AD
# integration tests (`test/integration/{enzyme,mooncake,reversediff}/main.jl`).
#
# Each test case carries a `tag::Symbol` identifying its source generator. Aggregate
# entry points return flat lists:
#   - `generate_ad_testcases()`      — every bijector-level AD case (ADTestCase).
#   - `generate_vector_testcases()`  — every distribution-level vector case (VectorTestCase).
# Each integration suite iterates these once with a local `is_broken(c)` predicate so the
# main file becomes one adtype list + one filter + two short loops.
#
# `ADTestCase` cases run via `run_ad_case` (gradient comparison against a FiniteDifferences
# reference). `VectorTestCase` cases run via `run_vector_case`: the structural checks come
# from `VectorBijectors.test_all` (DI-free, lives in `src/vector/test_utils.jl`), and when
# `adtypes` is non-empty, the AD-dependent checks (`test_all_ad`, defined here) run on top.
#
# Everything `VectorTestCase`-related lives in `test/vector_bijectors.jl`, which this file
# includes at the bottom — so a single `include("test_resources.jl")` is all integration
# tests need.

using Bijectors
using Bijectors: ordered
import Bijectors.VectorBijectors
using Bijectors.VectorBijectors:
    from_linked_vec, from_vec, linked_vec_length, to_linked_vec, to_vec
using DifferentiationInterface
import DifferentiationInterface as DI
using Distributions
const _D = Distributions
using FillArrays: Fill
using FiniteDifferences: central_fdm
using LinearAlgebra
using PDMats
using StableRNGs: StableRNG
using Test

# Seed used to construct randomised AD test inputs. Stable across CI runs so that a
# backend failure can be reproduced with the same `c.arg` and so all backends see
# identical inputs.
const TESTCASE_SEED = 23

_testcase_rng() = StableRNG(TESTCASE_SEED)

# Compact, readable name for a distribution. `Bijectors.VectorBijectors._name` already
# handles wrappers (Truncated, Censored, ReshapedDistribution, OrderStatistic, etc.); fall
# back to a stripped `nameof` for everything else so `VectorTestCase.name` doesn't carry
# the 1000-character `string(d)` form for nested products.
_case_name(d) = string(VectorBijectors._name(d))

# ===== Test case dispatch =====

# Per-tag generators live as methods of `_gen_testcases(::Val{:tag})`. The public
# `generate_testcases(Val(:tag))` wraps them and stamps `tag` onto every case so callers
# can filter a flat list of cases by source.
function _gen_testcases end
generate_testcases(t::Val{T}) where {T} = [_settag(c, T) for c in _gen_testcases(t)]

# ===== ADTestCase =====

struct ADTestCase
    name::String
    func::Function
    arg::Any
    tag::Symbol
end
ADTestCase(name::String, func, arg) = ADTestCase(name, func, arg, :_default)

_settag(c::ADTestCase, tag::Symbol) = ADTestCase(c.name, c.func, c.arg, tag)

const _AD_TAGS = (
    :veccorrbijector, :veccholeskybijector, :planarlayer, :pdvecbijector, :stackedbijector
)

"""
Return every bijector-level AD test case as a single flat list. Each case's `tag` field
identifies its source generator so integration suites can filter via `is_broken(c)`.
"""
generate_ad_testcases() = reduce(vcat, generate_testcases(Val(t)) for t in _AD_TAGS)

# ===== ADTestCase runner =====

const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))

function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
    @info "testing AD for function $f with $backend"
    ref_gradient = gradient(f, REF_BACKEND, x)
    ad_gradient = gradient(f, backend, x)
    @test isapprox(ad_gradient, ref_gradient; rtol=rtol, atol=atol)
end

function run_ad_case(c::ADTestCase, adtype; broken::Bool=false, rtol=1e-6, atol=1e-6)
    @testset "$(c.name)" begin
        if broken
            # Evaluate the comparison anyway under @test_broken: if `gradient` throws or
            # returns the wrong result the test stays broken; if the upstream bug is fixed
            # the case flips to "unexpectedly passing" and the maintainer gets a nudge.
            @info "testing (broken) AD for function $(c.func) with $adtype"
            @test_broken isapprox(
                gradient(c.func, adtype, c.arg),
                gradient(c.func, REF_BACKEND, c.arg);
                rtol=rtol,
                atol=atol,
            )
        else
            test_ad(c.func, adtype, c.arg; rtol=rtol, atol=atol)
        end
    end
end

# ===== Bijector-specific AD test cases =====

function _gen_testcases(::Val{:veccorrbijector})
    rng = _testcase_rng()
    cases = ADTestCase[]
    for d in (1, 2, 4)
        dist = LKJ(d, 2.0)
        b = bijector(dist)
        binv = inverse(b)
        x = rand(rng, dist)
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

function _gen_testcases(::Val{:veccholeskybijector})
    rng = _testcase_rng()
    cases = ADTestCase[]
    for d in (1, 2, 4), uplo in ('U', 'L')
        dist = LKJCholesky(d, 2.0, uplo)
        b = bijector(dist)
        binv = inverse(b)
        x = rand(rng, dist)
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
        push!(
            cases, ADTestCase("VecCholeskyBijector d=$d uplo=$uplo roundtrip", roundtrip, y)
        )
        push!(
            cases,
            ADTestCase("VecCholeskyBijector d=$d uplo=$uplo inverse", inverse_only, y),
        )
    end
    return cases
end

function _gen_testcases(::Val{:planarlayer})
    rng = _testcase_rng()
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
        ADTestCase("PlanarLayer logpdf vector input", f, randn(rng, 7)),
        ADTestCase("PlanarLayer logpdf matrix input", g, randn(rng, 11)),
        ADTestCase("PlanarLayer inverse logpdf vector input", finv, randn(rng, 7)),
        ADTestCase("PlanarLayer inverse logpdf matrix input", ginv, randn(rng, 11)),
    ]
end

function _gen_testcases(::Val{:pdvecbijector})
    rng = _testcase_rng()
    _topd(x) = x * x' + I
    d = 4
    b = Bijectors.PDVecBijector()
    binv = inverse(b)
    z = randn(rng, d, d)
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

function _gen_testcases(::Val{:stackedbijector})
    rng = _testcase_rng()
    dist1 = Dirichlet(4, 1.0)
    b1 = bijector(dist1)
    dist2 = LogNormal(0.0, 1.0)
    b2 = bijector(dist2)
    x1 = rand(rng, dist1)
    x2 = rand(rng, dist2)
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

# Pull in the VectorTestCase machinery + main-suite loop (guarded by GROUP).
include("vector/main.jl")
