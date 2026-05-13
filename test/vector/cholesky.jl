@testset "Cholesky" begin
    for c in generate_testcases(Val(:cholesky_dists))
        run_vector_case(c, adtypes)
    end
end
