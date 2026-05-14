using LinearAlgebra: Cholesky
using Test

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

# isapprox is not defined for some samples (specifically Cholesky and NTs), so we need to
# patch that
function _isapprox_safe(x, y; kwargs...)
    return isapprox(x, y; kwargs...)
end
function _isapprox_safe(x::NamedTuple{names}, y::NamedTuple{names}; kwargs...) where {names}
    for name in names
        if !_isapprox_safe(x[name], y[name]; kwargs...)
            return false
        end
    end
    return true
end
function _isapprox_safe(x::Cholesky, y::Cholesky; kwargs...)
    if x.uplo != y.uplo || size(x.UL) != size(y.UL)
        return false
    end
    return isapprox(x.UL, y.UL; kwargs...)
end

function test_all(
    d::D.Distribution;
    expected_zero_allocs=(),
    roundtrip_atol=1e-10,
    roundtrip_rtol=sqrt(eps()),
    test_in_support=(_get_value_support(d) <: D.Continuous),
    test_construction_type_stable=true,
)
    @info "Testing $(_name(d))"
    @testset "$(_name(d))" begin
        test_roundtrip(d)
        test_roundtrip_inverse(d, test_in_support, roundtrip_atol, roundtrip_rtol)
        test_type_stability(d, test_construction_type_stable)
        test_vec_lengths(d)
        test_optics(d)
        test_allocations(d, expected_zero_allocs)
        test_logjac(d)
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
                xnew = frvs(ffwd(x))
                # https://github.com/TuringLang/Bijectors.jl/issues/441
                if d isa D.JointOrderStatistics &&
                    (any(isnan, xnew) || !all(isfinite, xnew))
                    @warn "NaNs or Inf produced in roundtrip test for $(_name(d)), skipping isapprox test"
                else
                    @test _isapprox_safe(x, xnew)
                end
            end
        end
    end
end

"""
Test that from_linked_vec and to_linked_vec are inverses.

If `test_in_support`, then this additionally also tests that `from_linked_vec(dist)`
actually does map random vectors to the support of the distribution (i.e., `finv(y)` for
some random `y` is in the support of `d`).

If the distribution is not continuous, we can't really check this (in fact the test is quite
meaningless). So for discrete distributions this test is skipped. There are also other
occasions where we disable this because of e.g. numerical issues, like for LKJ.
"""
function test_roundtrip_inverse(d::D.Distribution, test_in_support, atol, rtol)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip inverse (linked): $(_name(d))" begin
        len = linked_vec_length(d)

        # Check that Distributions.jl can actually run insupport. Sometimes it can't, e.g.
        # with product_distribution(MvNormal(), MvNormal()), even though that function is
        # well-defined.
        x = rand(d)
        if test_in_support && (!hasmethod(D.insupport, Tuple{typeof(d),typeof(x)}))
            @info "No method for Distributions.insupport($(typeof(d)), $(typeof(x))), skipping in-support test"
            test_in_support = false
        end

        for _ in 1:100
            @testset let y = randn(len), d = d
                ffwd = to_linked_vec(d)
                frvs = from_linked_vec(d)
                x = frvs(y)
                if test_in_support
                    in_support = D.insupport(d, x)
                    if in_support isa Bool
                        @test in_support
                    elseif in_support isa AbstractArray{Bool,0}
                        # This happens sometimes:
                        # https://github.com/JuliaStats/Distributions.jl/issues/2026
                        @test in_support[]
                    else
                        # We _could_ just check `all(in_support)`, but I don't want to be
                        # caught off-guard by any bugs in the bijector's implementation that
                        # returns a wrong shape/type of `x`.
                        error(
                            "Distributions.insupport returned unexpected type: $(typeof(in_support))",
                        )
                    end
                end

                ynew = ffwd(x)
                if d isa D.JointOrderStatistics && (
                    any(isnan, x) ||
                    !all(isfinite, x) ||
                    any(isnan, ynew) ||
                    !all(isfinite, ynew)
                )
                    @warn "NaNs or Inf produced in roundtrip test for $(_name(d)), skipping isapprox test"
                else
                    @test _isapprox_safe(y, ynew; atol=atol, rtol=rtol)
                end
            end
        end
    end
end

"""
Test that the conversions to and from vector and linked vector forms for the given
distribution `d` are type-stable.

Sometimes the *creation* of the bijector itself is not type-stable (e.g. for product
distributions with heterogeneous components), but once the bijector is created, the
conversions should be type stable. To disable type stability checks for the construction,
set `test_construction_type_stable=false`.
"""
function test_type_stability(d::D.Distribution, test_construction_type_stable=true)
    x = rand(d)
    @testset "type stability: $(_name(d))" begin
        @testset let x = x, d = d
            if test_construction_type_stable
                @inferred to_vec(d)
                @inferred from_vec(d)
            end
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
    @testset "type stability (linked): $(_name(d))" begin
        @testset let x = x, d = d
            if test_construction_type_stable
                @inferred to_linked_vec(d)
                @inferred from_linked_vec(d)
            end
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

The companion check for `linked_optic_vec` requires an AD backend to compute the link
Jacobian and lives in `test/test_resources.jl` (called by the AD integration suites).
"""
function test_optics(d::D.Distribution)
    @testset "optic_vec: $(_name(d))" begin
        o = optic_vec(d)
        x = rand(d)
        v = to_vec(d)(x)
        for (optic, value) in zip(o, v)
            if optic !== nothing
                @test optic(x) == value
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
Test that the vectorisation conversions produce zero log-Jacobian (they are reshapes).

The companion check that the analytical *linked* log-Jacobian matches an AD-derived one
requires an AD backend and lives in `test/test_resources.jl` (called by the AD integration
suites).
"""
function test_logjac(d::D.Distribution)
    # Vectorisation logjacs should be zero because they are just reshapes.
    @testset "logjac: $(_name(d))" begin
        for _ in 1:100
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                y, logjac = with_logabsdet_jacobian(ffwd, x)
                @test _isapprox_safe(y, ffwd(x))
                @test iszero(logjac)
                frvs = from_vec(d)
                x_recon, logjac = with_logabsdet_jacobian(frvs, y)
                @test _isapprox_safe(x_recon, frvs(y))
                @test iszero(logjac)
            end
        end
    end
end
