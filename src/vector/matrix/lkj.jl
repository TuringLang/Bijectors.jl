# LKJ correlation matrices.
from_linked_vec(::D.LKJ) = inverse(B.VecCorrBijector())
to_linked_vec(::D.LKJ) = B.VecCorrBijector()
function linked_vec_length(d::D.LKJ)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
linked_optic_vec(d::D.LKJ) = fill(nothing, linked_vec_length(d))
