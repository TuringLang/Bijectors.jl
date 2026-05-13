@testset "TransformedDistributions" for c in generate_testcases(Val(:transformed_dists))
    run_vector_case(c, BASE_ADTYPES)
end
