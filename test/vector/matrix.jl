@testset "Matrix distributions" begin
    # Enzyme is tested separately in test/integration/enzyme.
    adtypes = [
        AutoReverseDiff(),
        AutoReverseDiff(; compile=true),
        AutoMooncake(),
        AutoMooncakeForward(),
    ]
    # ReverseDiff gives wrong results when differentiating through VecCorrBijector, so we
    # run LKJ with Mooncake only. https://github.com/TuringLang/Bijectors.jl/issues/434
    lkj_adtypes = [AutoMooncake(), AutoMooncakeForward()]

    for c in generate_testcases(Val(:matrix_dists))
        run_vector_case(c, adtypes)
    end
    for c in generate_testcases(Val(:lkj_matrix_dists))
        run_vector_case(c, lkj_adtypes)
    end
end
