@testset "Multivariates" begin
    for c in generate_testcases(Val(:multivariates))
        run_vector_case(c, adtypes)
    end
end
