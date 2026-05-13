@testset "TransformedDistributions" begin
    for c in generate_testcases(Val(:transformed_dists))
        run_vector_case(c, adtypes)
    end
end
