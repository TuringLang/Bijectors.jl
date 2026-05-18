# Can't test LKJCholesky(1, ...) because its linked vector is length-zero and
# DifferentiationInterface trips up with empty vectors.
const cholesky_dists = [
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

function _gen_testcases(::Val{:cholesky_dists})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in cholesky_dists]
end
