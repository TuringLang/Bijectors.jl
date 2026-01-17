# LKJ correlation matrices.

# TODO (possibly reimplement.
# struct FwdLKJ
#     size::Int
# end
# function with_logabsdet_jacobian(f::FwdLKJ, x::AbstractMatrix{T}) where {T<:Real}
#     n = f.size
#     y = zeros(T, div(n * (n - 1), 2))
#     return y, zero(T)
# end
# inverse(f::FwdLKJ) = InvLKJ(f.size)
#
# struct InvLKJ
#     size::Int
# end
# inverse(f::InvLKJ) = FwdLKJ(f.size)

# from_linked_vec(d::D.LKJ) = InvLKJ(first(size(d)))
# to_linked_vec(d::D.LKJ) = FwdLKJ(first(size(d)))

from_linked_vec(::D.LKJ) = inverse(B.VecCorrBijector())
to_linked_vec(::D.LKJ) = B.VecCorrBijector()
function linked_vec_length(d::D.LKJ)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
linked_optic_vec(d::D.LKJ) = fill(nothing, linked_vec_length(d))
