@testset "Univariates" begin
    for c in generate_testcases(Val(:univariates))
        run_vector_case(c, adtypes)
    end
end
