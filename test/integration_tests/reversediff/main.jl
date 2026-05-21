using ADTypes
using AbstractPPL: AbstractPPL
using Bijectors
using ChainRules: ChainRules
using DifferentiationInterface: DifferentiationInterface
using Distributions
using FillArrays: Fill
using ForwardDiff: ForwardDiff
using LinearAlgebra
using PDMats
using ReverseDiff: ReverseDiff
using Test

include(joinpath(@__DIR__, "..", "..", "test_resources.jl"))

const adtypes = [AutoReverseDiff(), AutoReverseDiff(; compile=true)]

# LKJ: https://github.com/TuringLang/Bijectors.jl/issues/434 (wrong forward jacobian
# through VecCorrBijector). JointOrderStatistics: https://github.com/JuliaDiff/ReverseDiff.jl/issues/43
# (setindex!).
const _BROKEN_VECTOR_TAGS = (:lkj_matrix_dists, :order_joint)

function vector_broken_adtypes(c::VectorTestCase)
    return c.tag in _BROKEN_VECTOR_TAGS ? adtypes : ADTypes.AbstractADType[]
end

@testset "ReverseDiff bijector AD" begin
    for c in generate_ad_testcases(), adtype in adtypes
        run_ad_case(c, adtype)
    end
end

@testset "ReverseDiff vector test_all" begin
    for c in generate_vector_testcases()
        run_vector_case(c, adtypes; broken_adtypes=vector_broken_adtypes(c))
    end
end
