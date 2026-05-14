# Main-suite execution of the non-AD `VectorBijectors.test_all` checks. Included from
# `test/runtests.jl` when `GROUP` is `All`, `Vector`, or `VectorProduct`. Requires
# `test_resources.jl` (case definitions + runner) to have been included already.
#
# `Vector` covers everything except product distributions, `VectorProduct` covers products
# only, `All` runs both.

let
    product_only_tags = (:products, :nested_product_namedtuple, :type_unstable_products)
    selected_tags = if GROUP == "Vector"
        Tuple(t for t in _VECTOR_TAGS if t ∉ product_only_tags)
    elseif GROUP == "VectorProduct"
        product_only_tags
    else
        _VECTOR_TAGS
    end

    @testset "VectorBijectors test_all" begin
        for c in generate_vector_testcases()
            c.tag in selected_tags || continue
            run_vector_case(c)
        end
    end
end
