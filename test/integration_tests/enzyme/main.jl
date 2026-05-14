using ADTypes
using Bijectors
using DifferentiationInterface
using Distributions
using Enzyme: Enzyme, set_runtime_activity, Forward, Reverse, Const
using EnzymeTestUtils: test_forward, test_reverse
using FillArrays: Fill
using FiniteDifferences
using ForwardDiff: ForwardDiff
using LinearAlgebra
using PDMats
using Test

include(joinpath(@__DIR__, "..", "..", "test_resources.jl"))

# `set_runtime_activity` is empirically required for VecCholesky / Stacked / LKJCholesky /
# products. Applying it uniformly (rather than per-tag) is the conservative form that works
# everywhere — we lose the assertion "this case happens to work without runtime_activity"
# in exchange for a single adtype list across all cases. `function_annotation=Const` pins
# DI's default to match what these tests were originally validated against.
const adtypes = [
    AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

# Enzyme cannot differentiate through triple-nested tuple-of-products (e.g.
# `product_distribution(p1t, p1t, p1t)`); identify them structurally.
function _enzyme_failing_product(d)
    d isa Distributions.ProductDistribution || return false
    d.dists isa Tuple || return false
    return first(d.dists) isa Union{Distributions.Product,Distributions.ProductDistribution}
end

function is_broken(c::Union{VectorTestCase,ADTestCase})
    c isa ADTestCase && return false
    # `reshape(Beta(2,2), (1,1,1,1,1))` hits https://github.com/EnzymeAD/Enzyme.jl/issues/2987
    # on Julia 1.10 (Reverse mode); mark the whole case broken on 1.10 rather than splitting
    # the adtype list per case.
    c.tag === :reshaped_beta_special && VERSION < v"1.11-" && return true
    # Triple-nested products: Enzyme bails on the activity inference.
    c.tag === :type_unstable_products && _enzyme_failing_product(c.dist) && return true
    return false
end

# This entire test suite is broken on 1.11.
#
# https://github.com/EnzymeAD/Enzyme.jl/issues/2121
# https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
#
# The fix to this needs to be made in Julia itself: it seems that this has already been done
# in https://github.com/JuliaLang/llvm-project/pull/49 although whether this will be
# incorporated into the built Julia version itself seems unclear. See
# https://github.com/JuliaLang/julia/pull/59521#issuecomment-3300480633.
#
# If this does not end up being backported to 1.11, then we may have to permanently skip
# these tests.
#
# On another note: Ideally we'd use `@test_throws`. However, that doesn't work because
# `test_forward` itself calls `@test`, and the error is captured by that `@test`, not our
# `@test_throws`. Consequently `@test_throws` doesn't actually see any error. Weird Julia
# behaviour.
@static if VERSION < v"1.11"
    @testset "Enzyme: Bijectors.find_alpha" begin
        x = randn()
        y = expm1(randn())
        z = randn()

        @testset "forward" begin
            @testset for RT in (Const, Enzyme.Duplicated, Enzyme.DuplicatedNoNeed),
                Tx in (Const, Enzyme.Duplicated),
                Ty in (Const, Enzyme.Duplicated),
                Tz in (Const, Enzyme.Duplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            @testset for RT in
                         (Const, Enzyme.BatchDuplicated, Enzyme.BatchDuplicatedNoNeed),
                Tx in (Const, Enzyme.BatchDuplicated),
                Ty in (Const, Enzyme.BatchDuplicated),
                Tz in (Const, Enzyme.BatchDuplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end
        end
        @testset "reverse" begin
            @testset for RT in (Const, Enzyme.Active),
                Tx in (Const, Enzyme.Active),
                Ty in (Const, Enzyme.Active),
                Tz in (Const, Enzyme.Active)

                test_reverse(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            # TODO: Test batch mode. Enzyme does not support all combinations of activities
            # currently:
            # https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2480468728
        end
    end
end

@testset "Enzyme bijector AD" begin
    for c in generate_ad_testcases(), adtype in adtypes
        run_ad_case(c, adtype; broken=is_broken(c))
    end
end

@testset "Enzyme vector test_all" begin
    for c in generate_vector_testcases()
        run_vector_case(c, adtypes; broken=is_broken(c))
    end
end
