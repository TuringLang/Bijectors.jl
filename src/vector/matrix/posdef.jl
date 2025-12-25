# Distributions over positive (semi)definite matrices.

const PDMatrixDistribution = Union{D.MatrixBeta,D.Wishart,D.InverseWishart}

from_linked_vec(::PDMatrixDistribution) = inverse(B.PDVecBijector())
to_linked_vec(::PDMatrixDistribution) = B.PDVecBijector()
function linked_vec_length(d::PDMatrixDistribution)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
linked_optic_vec(d::PDMatrixDistribution) = fill(nothing, linked_vec_length(d))
