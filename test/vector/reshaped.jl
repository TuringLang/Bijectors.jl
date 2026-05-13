@testset "Reshaped distributions" begin
    for c in generate_testcases(Val(:reshaped_dists))
        run_vector_case(c, NONENZYME_ADTYPES)
    end
    for c in generate_testcases(Val(:reshaped_beta_special))
        run_vector_case(c, NONENZYME_ADTYPES)
    end
end
