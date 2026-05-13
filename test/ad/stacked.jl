let cases = generate_testcases(Val(:stackedbijector))
    @testset "StackedBijector: $name" for (name, adtype) in TEST_ADTYPES
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end
