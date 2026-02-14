# MatrixNormal and MatrixTDist are trivial since all their components are already
# unconstrained.

const UnconsMatrixDist = Union{D.MatrixNormal,D.MatrixTDist}

to_linked_vec(::UnconsMatrixDist) = Vec()
from_linked_vec(d::UnconsMatrixDist) = Reshape(size(d))
linked_vec_length(d::UnconsMatrixDist) = prod(size(d))
function linked_optic_vec(d::UnconsMatrixDist)
    return map(c -> AbstractPPL.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end
