@testset "VecCorrBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    for c in generate_testcases(Val(:veccorrbijector))
        run_ad_case(c, adtype)
    end
end

@testset "VecCholeskyBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    for c in generate_testcases(Val(:veccholeskybijector))
        run_ad_case(c, adtype)
    end
end
