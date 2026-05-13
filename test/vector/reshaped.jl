@testset "Reshaped distributions" begin
    # Enzyme is tested separately in test/integration/enzyme.
    adtypes = [
        AutoReverseDiff(),
        AutoReverseDiff(; compile=true),
        AutoMooncake(),
        AutoMooncakeForward(),
    ]
    for c in generate_testcases(Val(:reshaped_dists))
        run_vector_case(c, adtypes)
    end
    for c in generate_testcases(Val(:reshaped_beta_special))
        run_vector_case(c, adtypes)
    end
end
