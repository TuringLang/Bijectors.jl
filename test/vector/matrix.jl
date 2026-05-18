const _matrix_ν = 5
const _matrix_M = [1 2 3; 4 5 6]
const _matrix_Σ = PDMats.PDMat([1 0.5; 0.5 1])
const _matrix_Ω = PDMats.PDMat([1 0.3 0.2; 0.3 1 0.4; 0.2 0.4 1])

const matrix_dists = [
    MatrixNormal(2, 4),
    MatrixNormal(3, 5),
    MatrixTDist(_matrix_ν, _matrix_M, _matrix_Σ, _matrix_Ω),
    Wishart(7, Matrix{Float64}(I, 2, 2)),
    Wishart(7, Matrix{Float64}(I, 4, 4)),
    InverseWishart(7, Matrix{Float64}(I, 2, 2)),
    InverseWishart(7, Matrix{Float64}(I, 4, 4)),
]

const lkj_matrix_dists = [LKJ(3, 1.0), LKJ(7, 1.0)]

function _gen_testcases(::Val{:matrix_dists})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in matrix_dists]
end

# LKJ is split into its own tag because ReverseDiff gives wrong results when differentiating
# through VecCorrBijector (https://github.com/TuringLang/Bijectors.jl/issues/434), so it
# only runs in the Mooncake integration suite. Don't check `from_linked_vec(d)(randn(...))`
# support — numerical precision in the inverse bijector means diagonal entries are not
# exactly 1 (https://github.com/TuringLang/Bijectors.jl/issues/435).
function _gen_testcases(::Val{:lkj_matrix_dists})
    return [
        VectorTestCase(d; expected_zero_allocs=(), test_in_support=false) for
        d in lkj_matrix_dists
    ]
end
