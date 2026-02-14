# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# The AbstractMvNormal abstract type takes care of MvNormal and MvNormalCanon.
from_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
to_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
linked_vec_length(d::D.AbstractMvNormal) = length(d)
linked_optic_vec(d::D.AbstractMvNormal) = optic_vec(d)
