# Shared test resources used by `test/runtests.jl` (main suite, including the inline
# vector-loop) and the AD integration tests
# (`test/integration_tests/{enzyme,mooncake,reversediff}/main.jl`).
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

# ===== VectorTestCase =====

struct VectorTestCase{D<:Distributions.Distribution}
    name::String
    dist::D
    test_kwargs::NamedTuple
    tag::Symbol
end

function VectorTestCase(name::String, dist::Distributions.Distribution; kwargs...)
    return VectorTestCase(name, dist, NamedTuple(kwargs), :_default)
end
function VectorTestCase(dist::Distributions.Distribution; kwargs...)
    return VectorTestCase(_case_name(dist), dist; kwargs...)
end

_settag(c::VectorTestCase, tag::Symbol) = VectorTestCase(c.name, c.dist, c.test_kwargs, tag)

const _VECTOR_TAGS = (
    :univariates,
    :multivariates,
    :matrix_dists,
    :lkj_matrix_dists,
    :cholesky_dists,
    :reshaped_dists,
    :reshaped_beta_special,
    :transformed_dists,
    :order_orderstatistic,
    :order_joint,
    :order_ordered,
    :products,
    :nested_product_namedtuple,
    :type_unstable_products,
)

"""
Return every distribution-level vector test case as a single flat list. Each case's `tag`
field identifies its source generator so integration suites can filter via `is_broken(c)`.
"""
generate_vector_testcases() = reduce(vcat, generate_testcases(Val(t)) for t in _VECTOR_TAGS)

# ===== VectorTestCase runner =====

function run_vector_case(c::VectorTestCase, adtypes=DI.AbstractADType[]; broken::Bool=false)
    if broken
        # `VectorBijectors.test_all` runs many internal `@test`s and doesn't return a
        # single pass/fail, so we mark broken cases with a bare `@test_broken false`
        # rather than running test_all and trying to capture every internal result.
        @testset "$(c.name)" begin
            @test_broken false
        end
        return nothing
    end
    VectorBijectors.test_all(c.dist; c.test_kwargs...)
    if !isempty(adtypes)
        test_all_ad(c.dist, adtypes)
    end
    return nothing
end

# ===== AD-dependent vector test helpers =====
#
# These pull in DifferentiationInterface and so cannot live in `src/vector/test_utils.jl`.
# Each AD integration test (test/integration_tests/<backend>/main.jl) calls `run_vector_case`
# with a non-empty `adtypes` list to exercise them.

const _REF_AD = DI.AutoForwardDiff()

# AD will give nonsense results at the limits of censored distributions (since the gradient
# is not well-defined), so we avoid generating samples that are exactly at the limits.
_rand_safe_ad(d::_D.Distribution) = rand(d)
function _rand_safe_ad(d::_D.Censored)
    a, b = d.lower, d.upper
    while true
        x = rand(d)
        if x != a && x != b
            return x
        end
    end
end

# When testing logjac for distributions where `vec_length(d) != linked_vec_length(d)`, if
# we naively try to compute the logjacobian of the transformation from vector to linked
# vector form (or vice versa), it will error because the dimensions don't match (i.e., the
# Jacobian is not square). See
# https://turinglang.org/Bijectors.jl/stable/defining_examples/#Stereographic-projection
# for an example of how to work around this issue.
# Here we define a function which converts a sample from `d` to a vector of length
# `linked_vec_length(d)` but does NOT perform linking. This allows us to compute the
# Jacobian from `to_vec_for_logjac_test(d)(x)` to `to_linked_vec(d)(x)`, which will be
# square. The fallback definition is just to_vec(d), but we can overload this for specific
# distributions.
to_vec_for_logjac_test(d::_D.Distribution) = to_vec(d)
from_vec_for_logjac_test(d::_D.Distribution) = from_vec(d)
to_vec_for_logjac_test(::Union{_D.Dirichlet,_D.MvLogitNormal}) = x -> x[1:(end - 1)]
from_vec_for_logjac_test(::Union{_D.Dirichlet,_D.MvLogitNormal}) = y -> vcat(y, 1 - sum(y))
function to_vec_for_logjac_test(
    d::Union{<:_D.ProductDistribution,<:_D.ProductNamedTupleDistribution}
)
    return VectorBijectors._make_transform(
        d.dists,
        to_vec_for_logjac_test,
        linked_vec_length,
        VectorBijectors.ProductVecTransform,
    )
end
function from_vec_for_logjac_test(
    d::Union{<:_D.ProductDistribution,<:_D.ProductNamedTupleDistribution}
)
    return VectorBijectors._make_transform(
        d.dists,
        from_vec_for_logjac_test,
        linked_vec_length,
        VectorBijectors.ProductVecInvTransform,
    )
end
function to_vec_for_logjac_test(
    ::_D.ReshapedDistribution{
        <:Any,<:_D.ValueSupport,<:Union{_D.Dirichlet,_D.MvLogitNormal}
    },
)
    return x -> vec(x)[1:(end - 1)]
end
function from_vec_for_logjac_test(
    d::_D.ReshapedDistribution{
        <:Any,<:_D.ValueSupport,<:Union{_D.Dirichlet,_D.MvLogitNormal}
    },
)
    return y -> reshape(vcat(y, 1 - sum(y)), size(d))
end
struct CholeskyToVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyToVecForLogjac)(x::Cholesky{T}) where {T<:Number}
    indices = VectorBijectors._get_cartesian_indices(c.n, c.uplo)
    vec_len = div(c.n * (c.n - 1), 2)
    xvec = Vector{T}(undef, vec_len)
    idx = 1
    for (i, j) in indices
        if i != j
            xvec[idx] = x.UL[i, j]
            idx += 1
        end
    end
    return xvec
end
to_vec_for_logjac_test(d::_D.LKJCholesky) = CholeskyToVecForLogjac(first(size(d)), d.uplo)
struct CholeskyFromVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyFromVecForLogjac)(xvec::AbstractVector{T}) where {T<:Number}
    indices = VectorBijectors._get_cartesian_indices(c.n, c.uplo)
    x = if c.uplo == 'U'
        Cholesky(UpperTriangular(zeros(T, c.n, c.n)))
    else
        Cholesky(LowerTriangular(zeros(T, c.n, c.n)))
    end
    idx = 1
    for (i, j) in indices
        if i != j
            x.UL[i, j] = xvec[idx]
            idx += 1
        end
    end
    for i in 1:(c.n)
        sum_sq = if c.uplo == 'U'
            sum(abs2, x.UL[:, i])
        else
            sum(abs2, x.UL[i, :])
        end
        x.UL[i, i] = sqrt(one(T) - sum_sq)
    end
    return x
end
function from_vec_for_logjac_test(d::_D.LKJCholesky)
    return CholeskyFromVecForLogjac(first(size(d)), d.uplo)
end

function to_vec_for_logjac_test(d::_D.ReshapedDistribution)
    return rx -> begin
        x = VectorBijectors._reshape_or_only(rx, size(d.dist))
        return to_vec_for_logjac_test(d.dist)(x)
    end
end
function from_vec_for_logjac_test(d::_D.ReshapedDistribution)
    return yvec -> begin
        x = from_vec_for_logjac_test(d.dist)(yvec)
        return VectorBijectors._reshape_or_only(x, size(d))
    end
end

# Positive (semi)definite matrix distributions are symmetric, so vectorise the
# lower-triangular part.
function to_vec_for_logjac_test(d::Union{_D.Wishart,_D.InverseWishart})
    n = first(size(d))
    return x -> begin
        vec_len = div(n * (n + 1), 2)
        xvec = zeros(eltype(x), vec_len)
        idx = 1
        for i in 1:n, j in 1:i
            xvec[idx] = x[i, j]
            idx += 1
        end
        return xvec
    end
end
function from_vec_for_logjac_test(d::Union{_D.Wishart,_D.InverseWishart})
    n = first(size(d))
    return xvec -> begin
        x = zeros(eltype(xvec), n, n)
        idx = 1
        for i in 1:n, j in 1:i
            x[i, j] = xvec[idx]
            x[j, i] = xvec[idx]
            idx += 1
        end
        return x
    end
end

# Correlation matrices: symmetric with all-ones diagonal.
function to_vec_for_logjac_test(d::_D.LKJ)
    n = first(size(d))
    return x -> begin
        vec_len = div(n * (n - 1), 2)
        xvec = zeros(eltype(x), vec_len)
        idx = 1
        for i in 1:n, j in 1:(i - 1)
            xvec[idx] = x[i, j]
            idx += 1
        end
        return xvec
    end
end
function from_vec_for_logjac_test(d::_D.LKJ)
    n = first(size(d))
    return xvec -> begin
        x = ones(eltype(xvec), n, n)
        idx = 1
        for i in 1:n, j in 1:(i - 1)
            x[i, j] = xvec[idx]
            x[j, i] = xvec[idx]
            idx += 1
        end
        return x
    end
end

"""
Test that the optics produced by `linked_optic_vec` line up with the Jacobian structure
of the link transform.
"""
function test_linked_optic(d::_D.Distribution)
    @testset "linked_optic_vec: $(VectorBijectors._name(d))" begin
        x = rand(d)
        xvec = to_vec(d)(x)
        yvec = to_linked_vec(d)(x)
        J = DI.jacobian(to_linked_vec(d) ∘ from_vec(d), _REF_AD, xvec)
        o = VectorBijectors.optic_vec(d)
        lo = VectorBijectors.linked_optic_vec(d)
        for i in 1:length(yvec)
            linked_optic = lo[i]
            if linked_optic !== nothing
                nonzero_index = findfirst(j -> o[j] == linked_optic, 1:length(xvec))
                if nonzero_index === nothing
                    error("linked_optic_vec produced an optic not found in optic_vec")
                end
                for j in 1:length(xvec)
                    if j != nonzero_index
                        @test iszero(J[i, j])
                    end
                end
            end
        end
    end
end

"""
Test that the analytical linked log-Jacobians match AD-derived log-Jacobians for `d`.
"""
function test_linked_logjac(d::_D.Distribution, atol, rtol)
    @testset "logjac (linked): $(VectorBijectors._name(d))" begin
        for _ in 1:100
            x = _rand_safe_ad(d)

            @testset let x = x, d = d
                # Sanity: to_vec_for_logjac_test and from_vec_for_logjac_test are inverses.
                @test VectorBijectors._isapprox_safe(
                    x,
                    from_vec_for_logjac_test(d)(to_vec_for_logjac_test(d)(x));
                    atol=atol,
                    rtol=rtol,
                )
            end

            @testset let x = x, d = d
                # Forward
                xvec = to_vec(d)(x)
                ffwd = to_linked_vec(d) ∘ from_vec(d)
                y, vbt_logjac = with_logabsdet_jacobian(ffwd, xvec)
                @test VectorBijectors._isapprox_safe(y, ffwd(xvec); atol=atol, rtol=rtol)
                ad_xvec = to_vec_for_logjac_test(d)(x)
                ad_ffwd = to_linked_vec(d) ∘ from_vec_for_logjac_test(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_ffwd, _REF_AD, ad_xvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end

            @testset let x = x, d = d
                # Reverse
                yvec = to_linked_vec(d)(x)
                vbt_frvs = to_vec(d) ∘ from_linked_vec(d)
                x, vbt_logjac = with_logabsdet_jacobian(vbt_frvs, yvec)
                @test VectorBijectors._isapprox_safe(
                    x, vbt_frvs(yvec); atol=atol, rtol=rtol
                )
                ad_frvs = to_vec_for_logjac_test(d) ∘ from_linked_vec(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_frvs, _REF_AD, yvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end
        end
    end
end

"""
Test that each AD backend in `adtypes` produces the same Jacobian / log-abs-det Jacobian
gradient as the ForwardDiff reference for the link and inverse-link transforms of `d`.
"""
function test_ad_distribution(d::_D.Distribution, adtypes, atol, rtol)
    # Mooncake refuses to differentiate identity transforms over discrete distributions,
    # and Enzyme errors with a Const annotation mismatch — filter both out for discrete d.
    adtypes = if d isa _D.Distribution{<:Any,_D.Discrete}
        filter(adtypes) do adtype
            !(
                adtype isa DI.AutoMooncake ||
                adtype isa DI.AutoMooncakeForward ||
                adtype isa DI.AutoEnzyme
            )
        end
    else
        adtypes
    end

    @testset "AD forward: $(VectorBijectors._name(d))" begin
        x = _rand_safe_ad(d)
        xvec = to_vec(d)(x)
        ffwd = to_linked_vec(d) ∘ from_vec(d)
        ref_jac = DI.jacobian(ffwd, _REF_AD, xvec)

        ladj(xvec) = last(with_logabsdet_jacobian(ffwd, xvec))
        ref_grad_ladj = DI.gradient(ladj, _REF_AD, xvec)

        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(ffwd, adtype, xvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
            @testset let x = x, adtype = adtype, d = d
                ad_grad_ladj = DI.gradient(ladj, adtype, xvec)
                @test ref_grad_ladj ≈ ad_grad_ladj atol = atol rtol = rtol
            end
        end
    end

    @testset "AD reverse: $(VectorBijectors._name(d))" begin
        x = _rand_safe_ad(d)
        yvec = to_linked_vec(d)(x)
        frvs = to_vec(d) ∘ from_linked_vec(d)
        ref_jac = DI.jacobian(frvs, _REF_AD, yvec)

        ladj(yvec) = last(with_logabsdet_jacobian(frvs, yvec))
        ref_grad_ladj = DI.gradient(ladj, _REF_AD, yvec)

        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(frvs, adtype, yvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
            @testset let x = x, adtype = adtype, d = d
                ad_grad_ladj = DI.gradient(ladj, adtype, yvec)
                @test ref_grad_ladj ≈ ad_grad_ladj atol = atol rtol = rtol
            end
        end
    end
end

"""
Run the three AD-dependent vector tests (`linked_optic_vec`, AD-vs-analytical linked
log-Jacobian, and direct AD differentiation) for `d` against `adtypes`.
"""
function test_all_ad(d::_D.Distribution, adtypes; atol=1e-10, rtol=sqrt(eps()))
    test_linked_optic(d)
    test_linked_logjac(d, atol, rtol)
    test_ad_distribution(d, adtypes, atol, rtol)
    return nothing
end

# ===== Per-category vector generators =====

include("vector/univariate.jl")
include("vector/multivariate.jl")
include("vector/matrix.jl")
include("vector/cholesky.jl")
include("vector/reshaped.jl")
include("vector/transformed.jl")
include("vector/order.jl")
include("vector/product.jl")
