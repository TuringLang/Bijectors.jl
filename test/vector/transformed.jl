const transformed_dists = [
    transformed(Normal(), exp),
    transformed(Beta(2, 3), Bijectors.Logit(0.0, 1.0)),
    transformed(Gamma(2, 1), elementwise(log)),
    transformed(product_distribution(fill(Beta(2, 2), 4)), elementwise(exp)),
    transformed(MvNormal(zeros(3), I), Bijectors.Scale(2.0)),
    transformed(Dirichlet([1.0, 2.0, 3.0])),
    transformed(MvLogNormal(zeros(2), I), elementwise(log)),
    transformed(MatrixNormal(zeros(2, 3), I(2), I(3)), elementwise(exp)),
]

function _gen_testcases(::Val{:transformed_dists})
    return [VectorTestCase(d; test_in_support=false) for d in transformed_dists]
end
