@testset "Multivariates" for c in generate_testcases(Val(:multivariates))
    run_vector_case(c, NONENZYME_ADTYPES)
end
