# MatrixNormal is trivial since all its components are already unconstrained.
to_linked_vec(::D.MatrixNormal) = Vec()
from_linked_vec(d::D.MatrixNormal) = Reshape(size(d))
linked_vec_length(d::D.MatrixNormal) = prod(size(d))
function linked_optic_vec(d::D.MatrixNormal)
    return map(c -> AbstractPPL.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end
