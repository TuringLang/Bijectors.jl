# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# AbstractMvNormal takes care of MvNormal and MvNormalCanon.
from_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
to_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
linked_vec_length(d::D.AbstractMvNormal) = length(d.μ)
