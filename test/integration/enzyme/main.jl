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

# Enzyme adtype flavours. `set_runtime_activity` is the load-bearing flag — empirically
# required for VecCholesky / Stacked / LKJCholesky / products, where Enzyme's compile-time
# activity inference fails. `function_annotation=Const` is set explicitly to pin DI's
# default. Some tag groups don't need either flag; flavours are kept separate to match the
# minimal configuration each test was originally validated against.
const bijector_backends = [
    ("EnzymeForward", AutoEnzyme(; mode=Forward)),
    ("EnzymeReverse", AutoEnzyme(; mode=Reverse)),
]
const runtime_const_backends = [
    AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]
const default_backends = [
    AutoEnzyme(; mode=Forward, function_annotation=Const),
    AutoEnzyme(; mode=Reverse, function_annotation=Const),
]
const joint_order_backends = [AutoEnzyme(; mode=Forward), AutoEnzyme(; mode=Reverse)]

# `enzyme_failures` from `main`: Enzyme cannot differentiate through triple-nested
# tuple-of-products (e.g. `product_distribution(p1t, p1t, p1t)`). Identify them
# structurally — `ProductDistribution` whose `.dists` is a `Tuple` of `Product` /
# `ProductDistribution` components.
function _enzyme_failing_product(d)
    d isa Distributions.ProductDistribution || return false
    d.dists isa Tuple || return false
    return first(d.dists) isa Union{Distributions.Product,Distributions.ProductDistribution}
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

# Bijector-level AD tests. The bare `AutoEnzyme(; mode=Forward)` / `mode=Reverse` form for
# VecCorrBijector/PlanarLayer/PDVecBijector matches the original integration test on
# `main`; VecCholeskyBijector and StackedBijector ran via TEST_ADTYPES on `main`, which
# used set_runtime_activity + Const.
let
    corr_cases = generate_testcases(Val(:veccorrbijector))
    planar_cases = generate_testcases(Val(:planarlayer))
    pd_cases = generate_testcases(Val(:pdvecbijector))
    @testset "$backend" for (backend, adtype) in bijector_backends
        @testset "VecCorrBijector" for c in corr_cases
            run_ad_case(c, adtype)
        end
        @testset "PlanarLayer" for c in planar_cases
            run_ad_case(c, adtype)
        end
        @testset "PDVecBijector" for c in pd_cases
            run_ad_case(c, adtype)
        end
    end
end

let cases = generate_testcases(Val(:veccholeskybijector))
    @testset "VecCholeskyBijector: $adtype" for adtype in runtime_const_backends
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end

let cases = generate_testcases(Val(:stackedbijector))
    @testset "StackedBijector: $adtype" for adtype in runtime_const_backends
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end

# Distribution-level `test_all` coverage moved from test/vector/*.jl.
@testset "Univariates" begin
    for c in generate_testcases(Val(:univariates))
        run_vector_case(c, default_backends)
    end
end

@testset "Multivariates" begin
    for c in generate_testcases(Val(:multivariates))
        run_vector_case(c, default_backends)
    end
end

# LKJ matrix dists ran with Mooncake only on `main`, so they have no Enzyme coverage.
@testset "Matrix distributions" begin
    for c in generate_testcases(Val(:matrix_dists))
        run_vector_case(c, default_backends)
    end
end

@testset "Cholesky" begin
    for c in generate_testcases(Val(:cholesky_dists))
        run_vector_case(c, runtime_const_backends)
    end
end

@testset "Order statistics" begin
    for c in generate_testcases(Val(:order_orderstatistic))
        run_vector_case(c, default_backends)
    end
    for c in generate_testcases(Val(:order_joint))
        run_vector_case(c, joint_order_backends)
    end
    for c in generate_testcases(Val(:order_ordered))
        run_vector_case(c, default_backends)
    end
end

@testset "Reshaped distributions" begin
    for c in generate_testcases(Val(:reshaped_dists))
        run_vector_case(c, default_backends)
    end
    # reshape(Beta(2, 2), (1, 1, 1, 1, 1)) hit
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2987 on Julia 1.10 — Enzyme Reverse
    # fails there, so on 1.10 we run only the Forward backend.
    beta_backends = VERSION >= v"1.11-" ? default_backends : default_backends[1:1]
    for c in generate_testcases(Val(:reshaped_beta_special))
        run_vector_case(c, beta_backends)
    end
end

@testset "TransformedDistributions" begin
    for c in generate_testcases(Val(:transformed_dists))
        run_vector_case(c, default_backends)
    end
end

@testset "Product distributions" begin
    for c in generate_testcases(Val(:products))
        run_vector_case(c, runtime_const_backends)
    end
    for c in generate_testcases(Val(:nested_product_namedtuple))
        run_vector_case(c, runtime_const_backends)
    end
    for c in generate_testcases(Val(:type_unstable_products))
        # Mark known Enzyme failures broken so they appear as broken in the test report
        # rather than being silently skipped. (VectorTestCase's broken flag is a marker;
        # run_vector_case does not invoke test_all in that branch.)
        if _enzyme_failing_product(c.dist)
            c = VectorTestCase(c.name, c.dist, c.test_kwargs, true)
        end
        run_vector_case(c, runtime_const_backends)
    end
end
