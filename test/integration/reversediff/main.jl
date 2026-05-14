using ADTypes
using Bijectors
using DifferentiationInterface
using Distributions
using FillArrays: Fill
using FiniteDifferences
using ForwardDiff: ForwardDiff
using LinearAlgebra
using PDMats
using ReverseDiff: ReverseDiff
using Test

include(joinpath(@__DIR__, "..", "..", "test_resources.jl"))

const adtypes = [AutoReverseDiff(), AutoReverseDiff(; compile=true)]

# ReverseDiff gives wrong results through VecCorrBijector (LKJ matrix dists,
# https://github.com/TuringLang/Bijectors.jl/issues/434) and can't differentiate
# JointOrderStatistics due to the heavy setindex! usage
# (https://github.com/JuliaDiff/ReverseDiff.jl/issues/43).
const _BROKEN_TAGS = (:lkj_matrix_dists, :order_joint)

is_broken(c::Union{VectorTestCase,ADTestCase}) = c.tag in _BROKEN_TAGS

@testset "ReverseDiff bijector AD" begin
    for c in generate_ad_testcases(), adtype in adtypes
        run_ad_case(c, adtype; broken=is_broken(c))
    end
end

@testset "ReverseDiff vector test_all" begin
    for c in generate_vector_testcases()
        run_vector_case(c, adtypes; broken=is_broken(c))
    end
end
