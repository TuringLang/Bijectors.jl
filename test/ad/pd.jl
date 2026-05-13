let cases = generate_testcases(Val(:pdvecbijector))
    @testset "PDVecBijector: $name" for (name, adtype) in TEST_ADTYPES
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end
