# LKJ correlation matrices.

struct FwdLKJ
    size::Int
end
function with_logabsdet_jacobian(f::FwdLKJ, x::AbstractMatrix{T}) where {T<:Real}
    n = f.size
    y = zeros(T, div(n * (n - 1), 2))
    # TODO
    return y, zero(T)
end
inverse(f::FwdLKJ) = InvLKJ(f.size)

struct InvLKJ
    size::Int
end
inverse(f::InvLKJ) = FwdLKJ(f.size)

from_linked_vec(d::D.LKJ) = InvLKJ(first(size(d)))
to_linked_vec(d::D.LKJ) = FwdLKJ(first(size(d)))
function linked_vec_length(d::D.LKJ)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
linked_optic_vec(d::D.LKJ) = fill(nothing, linked_vec_length(d))
