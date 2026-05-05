# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# The AbstractMvNormal abstract type takes care of MvNormal and MvNormalCanon.
from_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
to_linked_vec(::D.AbstractMvNormal) = TypedIdentity()
linked_vec_length(d::D.AbstractMvNormal) = length(d)
linked_optic_vec(d::D.AbstractMvNormal) = optic_vec(d)

# NOTE: AbstractMvTDist is not formally exported from Distributions, but this is the only
# 'correct' place to put it
from_linked_vec(::D.AbstractMvTDist) = TypedIdentity()
to_linked_vec(::D.AbstractMvTDist) = TypedIdentity()
linked_vec_length(d::D.AbstractMvTDist) = length(d)
linked_optic_vec(d::D.AbstractMvTDist) = optic_vec(d)
