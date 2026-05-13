@testset "Univariates" for c in generate_testcases(Val(:univariates))
    run_vector_case(c, NONENZYME_ADTYPES)
end
