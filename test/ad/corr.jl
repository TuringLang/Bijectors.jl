let cases = generate_testcases(Val(:veccorrbijector))
    @testset "VecCorrBijector: $name" for (name, adtype) in TEST_ADTYPES
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end

let cases = generate_testcases(Val(:veccholeskybijector))
    @testset "VecCholeskyBijector: $name" for (name, adtype) in TEST_ADTYPES
        for c in cases
            run_ad_case(c, adtype)
        end
    end
end
