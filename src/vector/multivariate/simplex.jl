# This file contains transforms for unit simplices, i.e., vectors where all elements
# are non-negative and sum to one. Here we can directly use the original bijector
# implementation since it already satisfies the required interface.
const SIMPLEX_MULTIVARIATES = Union{D.Dirichlet,D.MvLogitNormal}
VectorBijectors.from_linked_vec(::SIMPLEX_MULTIVARIATES) = inverse(B.SimplexBijector())
VectorBijectors.to_linked_vec(::SIMPLEX_MULTIVARIATES) = B.SimplexBijector()
VectorBijectors.linked_vec_length(d::SIMPLEX_MULTIVARIATES) = length(d) - 1
function linked_optic_vec(d::SIMPLEX_MULTIVARIATES)
    return fill(nothing, linked_vec_length(d))
end
