@testset "Cholesky" for c in generate_testcases(Val(:cholesky_dists))
    run_vector_case(c, BASE_ADTYPES)
end
