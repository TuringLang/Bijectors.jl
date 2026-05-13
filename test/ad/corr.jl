@testset "VecCorrBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    test_veccorrbijector_ad(adtype)
end

@testset "VecCholeskyBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    test_veccholeskybijector_ad(adtype)
end
