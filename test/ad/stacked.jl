@testset "StackedBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    for c in generate_testcases(Val(:stackedbijector))
        run_ad_case(c, adtype)
    end
end
