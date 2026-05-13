@testset "Matrix distributions" begin
    for c in generate_testcases(Val(:matrix_dists))
        run_vector_case(c, BASE_ADTYPES)
    end
    # ReverseDiff gives wrong results through VecCorrBijector, so LKJ runs with Mooncake
    # only. https://github.com/TuringLang/Bijectors.jl/issues/434
    lkj_adtypes = [AutoMooncake(), AutoMooncakeForward()]
    for c in generate_testcases(Val(:lkj_matrix_dists))
        run_vector_case(c, lkj_adtypes)
    end
end
