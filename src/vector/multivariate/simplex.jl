# This file contains transforms for unit simplices, i.e., vectors where all elements
# are non-negative and sum to one. It uses the stick-breaking algorithm, which is 
# directly lifted from Bijectors, and was used in Stan up till v2.36.

# Here we can directly use the original bijector implementation since it already satisfies
# the required interface.

const InverseSimplexBijector = inverse(B.SimplexBijector())

for dist_type in [D.Dirichlet, D.MvLogitNormal]
    @eval begin
        VectorBijectors.from_linked_vec(::$dist_type) = B.SimplexBijector()
        VectorBijectors.to_linked_vec(::$dist_type) = InverseSimplexBijector()
        VectorBijectors.linked_vec_length(d::$dist_type) = length(d) - 1
    end
end
