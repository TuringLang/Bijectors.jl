@testset "StackedBijector: $backend_name" for (backend_name, adtype) in TEST_ADTYPES
    test_stackedbijector_ad(adtype)
end
