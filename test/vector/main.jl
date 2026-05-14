# Everything `VectorTestCase`-related: the struct, the per-tag generators (split across the
# sibling files in this directory), the runner, and the AD-dependent companion checks.
# Included from `test/test_resources.jl` (so the integration tests get it for free). The
# main-suite execution loop at the bottom is guarded by `GROUP` so integration tests skip
# it and provide their own.

# ===== Test case type =====

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

# ===== Tags and aggregate =====

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

# ===== Runner =====

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
# Each AD integration test (test/integration/<backend>/main.jl) calls `run_vector_case`
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

# ===== Per-tag generators (split across sibling files) =====

include("univariate.jl")
include("multivariate.jl")
include("matrix.jl")
include("cholesky.jl")
include("reshaped.jl")
include("transformed.jl")
include("order.jl")
include("product.jl")

# ===== Main-suite execution =====
#
# Runs only when this file is loaded from `test/runtests.jl` with a Vector* group.
# Integration tests include this file via `test_resources.jl` but never set `GROUP`, so
# they skip the loop and provide their own.
if @isdefined(GROUP) && GROUP in ("All", "Vector", "VectorProduct")
    let
        product_only_tags = (:products, :nested_product_namedtuple, :type_unstable_products)
        selected_tags = if GROUP == "Vector"
            Tuple(t for t in _VECTOR_TAGS if t ∉ product_only_tags)
        elseif GROUP == "VectorProduct"
            product_only_tags
        else
            _VECTOR_TAGS
        end

        @testset "VectorBijectors test_all" begin
            for c in generate_vector_testcases()
                c.tag in selected_tags || continue
                run_vector_case(c)
            end
        end
    end
end
