@testset "PDVecBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    for c in generate_testcases(Val(:pdvecbijector))
        run_ad_case(c, adtype)
    end
end
