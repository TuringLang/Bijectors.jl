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

const SHARED = joinpath(@__DIR__, "..", "..", "shared")
include(joinpath(SHARED, "ad_test_utils.jl"))
include(joinpath(SHARED, "ad_bijector_tests.jl"))
include(joinpath(SHARED, "vector_distributions.jl"))

# Enzyme adtype configurations. Each list matches a flavour previously used in a different
# place on `main` so the moved tests run against the exact same backend they did before:
# - `bijector_backends`: original test/integration/enzyme/main.jl
# - `runtime_const_backends`: test/runtests.jl::TEST_ADTYPES (test/ad/{corr,stacked}.jl)
#   and test/vector/{cholesky,product}.jl
# - `default_backends`: src/vector/test_utils.jl::default_adtypes (test/vector/
#   {univariate,multivariate,matrix,transformed,reshaped}.jl)
# - `joint_order_backends`: test/vector/order.jl::joint_test_adtypes
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
const reshaped_beta_pre_111_backends =
    [AutoEnzyme(; mode=Forward, function_annotation=Const)]

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
@testset "$backend" for (backend, adtype) in bijector_backends
    @testset "VecCorrBijector" test_veccorrbijector_ad(adtype)
    @testset "PlanarLayer" test_planarlayer_ad(adtype)
    @testset "PDVecBijector" test_pdvecbijector_ad(adtype)
end

@testset "VecCholeskyBijector: $adtype" for adtype in runtime_const_backends
    test_veccholeskybijector_ad(adtype)
end

@testset "StackedBijector: $adtype" for adtype in runtime_const_backends
    test_stackedbijector_ad(adtype)
end

# Distribution-level test_all coverage moved from test/vector/*.jl. The adtype list passed
# to each wrapper matches what `main` used for that file's Enzyme entries.
@testset "Univariates" test_univariates_with(default_backends)
@testset "Multivariates" test_multivariates_with(default_backends)
# LKJ ran with Mooncake only on `main`, so it has no Enzyme coverage to move.
@testset "Matrix distributions" test_matrix_dists_with(default_backends; lkj_adtypes=[])
@testset "Cholesky" test_cholesky_with(runtime_const_backends)
@testset "Order statistics" test_order_with(
    default_backends; joint_adtypes=joint_order_backends
)
@testset "Reshaped distributions" test_reshaped_with(
    default_backends; beta_reshape_adtypes_pre_111=reshaped_beta_pre_111_backends
)
@testset "TransformedDistributions" test_transformed_with(default_backends)
@testset "Product distributions" test_products_with(runtime_const_backends)
