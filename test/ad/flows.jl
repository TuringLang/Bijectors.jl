let cases = generate_testcases(Val(:planarlayer))
    @testset "PlanarLayer: $name" for (name, adtype) in TEST_ADTYPES
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end
