@testset "TransformedDistributions" for c in generate_testcases(Val(:transformed_dists))
    run_vector_case(c, NONENZYME_ADTYPES)
end
