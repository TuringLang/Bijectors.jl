@testset "Product distributions" begin
    for c in generate_testcases(Val(:products))
        run_vector_case(c, adtypes)
    end
    for c in generate_testcases(Val(:nested_product_namedtuple))
        run_vector_case(c, adtypes)
    end
    for c in generate_testcases(Val(:type_unstable_products))
        run_vector_case(c, adtypes)
    end
end
