const _m2 = MvNormal(zeros(2), I)
const _d2 = Dirichlet(ones(2))
const _p1t = product_distribution(Normal(), Beta(2, 2))
const _p2t = product_distribution(_m2, _d2)
const _p1a = product_distribution(fill(Beta(2, 2), 2))
const _p2a = product_distribution(fill(_d2, 2))

# Purposely chosen so that `vec_length` agrees but `linked_vec_length` differs.
const products = [
    product_distribution(Normal()),
    product_distribution(Normal(), Normal()),
    product_distribution(Normal(), Beta(2, 2)),
    product_distribution(Beta(2, 2), Exponential()),
    product_distribution(_m2, _d2),
    product_distribution(_m2, _d2, _m2, _d2),
    product_distribution(fill(Normal(), 2)),
    product_distribution(fill(Beta(2, 2), 2)),
    product_distribution([Uniform(0, 1), Uniform(1, 2), Uniform(2, 3)]),
    product_distribution(Fill(Uniform(1, 2), 2)),
    product_distribution(fill(Normal(), 2, 2)),
    product_distribution(Fill(Uniform(1, 2), 2, 2)),
    product_distribution(fill(_m2, 2, 2)),
    product_distribution(Fill(_m2, 2, 2)),
    product_distribution(fill(_d2, 2, 2)),
    product_distribution((a=Normal(), b=Beta(2, 2))),
    product_distribution((a=Normal(), b=Dirichlet(ones(2)))),
    product_distribution((a=Normal(), b=product_distribution(fill(Beta(2, 2), 2)))),
    product_distribution(fill(_p1t, 2)),
    product_distribution(fill(_p1t, 2, 2)),
    product_distribution(_p2t, _p2t, _p2t),
    product_distribution(fill(_p2t, 2)),
    product_distribution(fill(_p2t, 2, 2)),
    product_distribution(fill(_p1a, 2)),
    product_distribution(fill(_p1a, 2, 2)),
    product_distribution(_p2a, _p2a, _p2a),
    product_distribution(fill(_p2a, 2)),
    product_distribution(fill(_p2a, 2, 2)),
]

# On Julia 1.10 (and only 1.10), `@inferred to_vec(d)` fails for this case even though
# `@code_warntype to_vec(d)` is type stable. Almost certainly a Julia bug.
const nested_product_namedtuple = [
    product_distribution((a=Normal(), b=product_distribution((c=Normal(), d=Beta(2, 2)))))
]

# Heterogeneous arrays make bijector construction type unstable. The triple-nested tuple
# products (last two) hit Enzyme activity-inference limits; the Enzyme suite filters them.
const type_unstable_products = [
    product_distribution([Normal(), Beta(2, 2), Exponential()]),
    product_distribution([Normal() Beta(2, 2); Exponential() Uniform(-1, 1)]),
    product_distribution([_m2 _d2; _m2 _d2]),
    product_distribution(_p1t, _p1t, _p1t),
    product_distribution(_p1a, _p1a, _p1a),
]

function _gen_testcases(::Val{:products})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in products]
end

function _gen_testcases(::Val{:nested_product_namedtuple})
    return [
        VectorTestCase(
            d; expected_zero_allocs=(), test_construction_type_stable=(VERSION >= v"1.11-")
        ) for d in nested_product_namedtuple
    ]
end

function _gen_testcases(::Val{:type_unstable_products})
    return [
        VectorTestCase(d; expected_zero_allocs=(), test_construction_type_stable=false) for
        d in type_unstable_products
    ]
end
