using Test
using AbstractPPL: @varname
using LinearAlgebra: logabsdet, Cholesky, UpperTriangular, LowerTriangular
import DifferentiationInterface as DI

# Would like to use FiniteDifferences, but very easy to run into issues with
# https://juliadiff.org/FiniteDifferences.jl/latest/#Dealing-with-Singularities
const ref_adtype = DI.AutoForwardDiff()

const default_adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
]

_get_value_support(::D.Distribution{<:Any,VS}) where {VS<:D.ValueSupport} = VS

# Pretty-printing distributions. Otherwise things like MvNormal are super ugly.
_name(d::D.Distribution) = nameof(typeof(d))
_name(d::D.Censored) = "censored $(_name(d.uncensored)) [$(d.lower),$(d.upper)]"
function _name(d::D.Truncated)
    return "truncated $(_name(d.untruncated)) [$(d.lower),$(d.upper)]"
end
function _name(d::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:D.Distribution})
    return "reshaped $(_name(d.dist)) to size $(size(d))"
end
_name(d::D.OrderStatistic) = "order statistic $(_name(d.dist))"
function _name(d::D.JointOrderStatistics)
    return "joint order statistic $(_name(d.dist)) with length $(length(d))"
end

# AD will give nonsense results at the limits of censored distributions (since the gradient
# is not well-defined), so we avoid generating samples that are exactly at the limits.
_rand_safe_ad(d::D.Distribution) = rand(d)
_rand_safe_ad(d::D.Censored) = begin
    a, b = d.lower, d.upper
    while true
        x = rand(d)
        if x != a && x != b
            return x
        end
    end
end

# isapprox is not defined for some samples (specifically Cholesky), so we need to patch that
function _isapprox_safe(x, y; kwargs...)
    return isapprox(x, y; kwargs...)
end
function _isapprox_safe(x::Cholesky, y::Cholesky; kwargs...)
    if x.uplo != y.uplo || size(x.UL) != size(y.UL)
        return false
    end
    return isapprox(x.UL, y.UL, kwargs...)
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
to_vec_for_logjac_test(d::D.Distribution) = to_vec(d)
from_vec_for_logjac_test(d::D.Distribution) = from_vec(d)
to_vec_for_logjac_test(::Union{D.Dirichlet,D.MvLogitNormal}) = x -> x[1:(end - 1)]
from_vec_for_logjac_test(::Union{D.Dirichlet,D.MvLogitNormal}) = y -> vcat(y, 1 - sum(y))
function to_vec_for_logjac_test(
    ::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:Union{D.Dirichlet,D.MvLogitNormal}}
)
    return x -> vec(x)[1:(end - 1)]
end
function from_vec_for_logjac_test(
    d::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:Union{D.Dirichlet,D.MvLogitNormal}}
)
    return y -> reshape(vcat(y, 1 - sum(y)), size(d))
end
struct CholeskyToVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyToVecForLogjac)(x::Cholesky{T}) where {T<:Number}
    # Same as to_vec, but skip the diagonal entries.
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
to_vec_for_logjac_test(d::D.LKJCholesky) = CholeskyToVecForLogjac(first(size(d)), d.uplo)
struct CholeskyFromVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyFromVecForLogjac)(xvec::AbstractVector{T}) where {T<:Number}
    # Same as from_vec, but skip the diagonal entries, and reconstruct them
    # from the fact that the rows/columns are unit-norm.
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
        # x.UL[i, i] is still zero now, so we can compute the sum-of-squares
        # including it, before then calculating it
        sum_sq = if c.uplo == 'U'
            sum(abs2, x.UL[:, i])
        else
            sum(abs2, x.UL[i, :])
        end
        x.UL[i, i] = sqrt(one(T) - sum_sq)
    end
    return x
end
function from_vec_for_logjac_test(d::D.LKJCholesky)
    return CholeskyFromVecForLogjac(first(size(d)), d.uplo)
end

# These are positive (semi)definite matrix distributions, which are symmetric, so we will
# just vectorise the lower-triangular part.
function to_vec_for_logjac_test(d::Union{D.MatrixBeta,D.Wishart,D.InverseWishart})
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
function from_vec_for_logjac_test(d::Union{D.MatrixBeta,D.Wishart,D.InverseWishart})
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

# These are correlation matrices - they are symmetric and the diagonal is all ones
function to_vec_for_logjac_test(d::D.LKJ)
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
function from_vec_for_logjac_test(d::D.LKJ)
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

function test_all(
    d::D.Distribution;
    expected_zero_allocs=(),
    adtypes=default_adtypes,
    ad_atol=1e-10,
    ad_rtol=sqrt(eps()),
    roundtrip_atol=1e-10,
    roundtrip_rtol=sqrt(eps()),
)
    @info "Testing $(_name(d))"
    @testset "$(_name(d))" begin
        test_roundtrip(d)
        test_roundtrip_inverse(d, roundtrip_atol, roundtrip_rtol)
        test_type_stability(d)
        test_vec_lengths(d)
        test_optics(d)
        test_allocations(d, expected_zero_allocs)
        test_logjac(d, ad_atol, ad_rtol)
        test_ad(d, adtypes, ad_atol, ad_rtol)
    end
end

"""
Test that from_vec and to_vec are inverses, and likewise for from_linked_vec and
to_linked_vec. This test checks `x ≈ from_vec(d)(to_vec(d)(x))` for random samples `x ~ d`
(and likewise for the linked transforms).
"""
function test_roundtrip(d::D.Distribution)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip: $(_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                frvs = from_vec(d)
                @test _isapprox_safe(x, frvs(ffwd(x)))
            end
        end
    end
    @testset "roundtrip (linked): $(_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_linked_vec(d)
                frvs = from_linked_vec(d)
                @test _isapprox_safe(x, frvs(ffwd(x)))
            end
        end
    end
end

"""
Test that from_linked_vec and to_linked_vec are inverses, and that they actually
do map random vectors to the support of the distribution.
"""
function test_roundtrip_inverse(d::D.Distribution, atol, rtol)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip inverse (linked): $(_name(d))" begin
        len = linked_vec_length(d)
        for _ in 1:100
            @testset let y = randn(len), d = d
                ffwd = to_linked_vec(d)
                frvs = from_linked_vec(d)
                x = frvs(y)
                # If the distribution is not continuous, we can't really check this (in fact
                # the test is quite meaningless). So for discrete distributions this
                # basically only checks that the ffwd and frvs are inverses.
                if _get_value_support(d) <: D.Continuous
                    @test D.insupport(d, x)
                end
                @test y ≈ ffwd(x) atol = atol rtol = rtol
            end
        end
    end
end

"""
Test that the conversions to and from vector and linked vector forms for the given
distribution `d` are type-stable.
"""
function test_type_stability(d::D.Distribution)
    x = rand(d)
    @testset "type stability: $(_name(d))" begin
        @testset let x = x, d = d
            @inferred to_vec(d)
            @inferred from_vec(d)
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
    @testset "type stability (linked): $(_name(d))" begin
        @testset let x = x, d = d
            @inferred to_linked_vec(d)
            @inferred from_linked_vec(d)
            ffwd = to_linked_vec(d)
            frvs = from_linked_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
end

"""
Test that the optics produced by `optic_vec` for the given distribution `d` line up with the
values produced by `to_vec`.
"""
function test_optics(d::D.Distribution)
    @testset "optic_vec: $(_name(d))" begin
        o = optic_vec(d)
        x = rand(d)
        v = to_vec(d)(x)
        for (optic, value) in zip(o, v)
            @test optic(x) == value
        end
    end

    @testset "linked_optic_vec: $(_name(d))" begin
        # This is a lot harder to test. What we need to prove is that
        #       x = rand(d)
        #       lv = to_linked_vec(d)(x)
        #       lo = linked_optic_vec(d)
        #  then for each element of `lv, lo` we have that `lv[i]` depends only on lo[i](x)
        #  and not any other elements of `x`. Conceptually, this means that if we take the
        #  Jacobian of the link transform, row `i` should have nonzeros only in the columns
        #  corresponding to `lo[i]`. This is a bit finicky to do because `x` might not be a
        #  vector(!) so we need to flatten everything first, using `to_vec`.
        x = rand(d)
        xvec = to_vec(d)(x)
        yvec = to_linked_vec(d)(x)
        J = DI.jacobian(to_linked_vec(d) ∘ from_vec(d), ref_adtype, xvec)
        o = optic_vec(d)
        lo = linked_optic_vec(d)
        for i in 1:length(yvec)
            linked_optic = lo[i]
            if linked_optic !== nothing
                # If the optic is non-nothing, then it refers to a specific element
                # of x. That means that we should be able to find, which index of the
                # input `xvec` it corresponds to, by finding the index `j` where
                # `optic_vec(d)[j] === linked_optic`.
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
Test that the lengths of the vectors produced by the conversions to vector and linked
vector forms for the given distribution `d` match those reported by `vec_length` and
`linked_vec_length`.
"""
function test_vec_lengths(d::D.Distribution)
    @testset "vector lengths: $(_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_vec(d)(x)
                @test length(y) == vec_length(d)
            end
        end
    end
    @testset "vector lengths (linked): $(_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_linked_vec(d)(x)
                @test length(y) == linked_vec_length(d)
            end
        end
    end
end

"""
Test that the conversions to and from vector and linked vector forms for the given
distribution `d` do not cause any heap allocations for the functions specified in
`expected_zero_allocs`.
"""
function test_allocations(d::D.Distribution, expected_zero_allocs=())
    ALLOWED_FUNCTIONS = (to_vec, from_vec, to_linked_vec, from_linked_vec)
    if any(f -> !(f in ALLOWED_FUNCTIONS), expected_zero_allocs)
        throw(ArgumentError("expected_zero_allocs can only contain: $ALLOWED_FUNCTIONS"))
    end
    # For univariates, to_vec and to_linked_vec always cause allocations because they have
    # to create a new vector.
    # TODO: Generalise to multivariates etc
    x = rand(d)
    @testset "allocations: $(_name(d))" begin
        @testset let x = x, d = d
            if to_vec in expected_zero_allocs
                ffwd = to_vec(d)
                ffwd(x)
                @test (@allocations ffwd(x)) == 0
            end
            if from_vec in expected_zero_allocs
                yvec = to_vec(d)(x)
                frvs = from_vec(d)
                frvs(yvec)
                @test (@allocations frvs(yvec)) == 0
            end
        end
    end
    @testset "allocations (linked): $(_name(d))" begin
        @testset let x = x, d = d
            if to_linked_vec in expected_zero_allocs
                ffwd = to_linked_vec(d)
                ffwd(x)
                @test (@allocations ffwd(x)) == 0
            end
            if from_linked_vec in expected_zero_allocs
                yvec = to_linked_vec(d)(x)
                frvs = from_linked_vec(d)
                frvs(yvec)
                @test (@allocations frvs(yvec)) == 0
            end
        end
    end
end

"""
Test that the analytical log-Jacobians provided in this package are correct by comparing
against AD-calculated log-Jacobians for the given distribution `d`.
"""
function test_logjac(d::D.Distribution, atol, rtol)
    # Vectorisation logjacs should be zero because they are just reshapes.
    @testset "logjac: $(_name(d))" begin
        for _ in 1:100
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                @test iszero(last(with_logabsdet_jacobian(ffwd, x)))
                y = ffwd(x)
                frvs = from_vec(d)
                @test iszero(last(with_logabsdet_jacobian(frvs, y)))
            end
        end
    end

    # Link logjacs will not be zero, so we need to check against a chosen backend. Because
    # Jacobians need to map from vector to vector, here we test the transformation of the
    # vectorised form to the linked vectorised form via the original sample.
    @testset "logjac (linked): $(_name(d))" begin
        for _ in 1:100
            x = _rand_safe_ad(d)

            @testset let x = x, d = d
                # As a sanity check we should make sure that to_vec_for_logjac_test and
                # from_vec_for_logjac_test are inverses. If they aren't, then that brings
                # the entire testset into question!
                @test _isapprox_safe(
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
                vbt_logjac = last(with_logabsdet_jacobian(ffwd, xvec))
                # For the AD calculation we need to use to/from_vec_for_logjac_test instead,
                # to make sure that the Jacobian is square.
                ad_xvec = to_vec_for_logjac_test(d)(x)
                ad_ffwd = to_linked_vec(d) ∘ from_vec_for_logjac_test(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_ffwd, ref_adtype, ad_xvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end

            @testset let x = x, d = d
                # Reverse
                yvec = to_linked_vec(d)(x)
                vbt_frvs = to_vec(d) ∘ from_linked_vec(d)
                vbt_logjac = last(with_logabsdet_jacobian(vbt_frvs, yvec))
                # For the AD calculation we need to use to/from_vec_for_logjac_test instead,
                # to make sure that the Jacobian is square.
                ad_frvs = to_vec_for_logjac_test(d) ∘ from_linked_vec(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_frvs, ref_adtype, yvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end
        end
    end
end

"""
Test that various AD backends can differentiate the conversions to and from vector and
linked vector forms for the given distribution `d`.
"""
function test_ad(d::D.Distribution, adtypes::Vector{<:DI.AbstractADType}, atol, rtol)
    # If `d` is a discrete distribution, Mooncake refuses to differentiate through the
    # transforms (which are just identity transforms). Arguably, the other AD backends
    # probably should do the same, but they do actually return the right 'gradients' so we
    # can test them.
    adtypes = if d isa D.Distribution{<:Any,D.Discrete}
        filter(adtypes) do adtype
            !(adtype isa DI.AutoMooncake || adtype isa DI.AutoMooncakeForward)
        end
    else
        adtypes
    end

    @testset "AD forward: $(_name(d))" begin
        x = _rand_safe_ad(d)
        xvec = to_vec(d)(x)
        ffwd = to_linked_vec(d) ∘ from_vec(d)
        ref_jac = DI.jacobian(ffwd, ref_adtype, xvec)
        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(ffwd, adtype, xvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
        end
    end
    @testset "AD reverse: $(_name(d))" begin
        x = _rand_safe_ad(d)
        yvec = to_linked_vec(d)(x)
        frvs = to_vec(d) ∘ from_linked_vec(d)
        ref_jac = DI.jacobian(frvs, ref_adtype, yvec)
        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(frvs, adtype, yvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
        end
    end
end
