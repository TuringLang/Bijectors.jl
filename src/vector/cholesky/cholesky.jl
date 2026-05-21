using LinearAlgebra: LinearAlgebra as LA

function _get_cartesian_indices(n::Int, uplo::Char)
    if uplo == 'U'
        return [(i, j) for j in 1:n for i in 1:j]
    else
        return [(i, j) for j in 1:n for i in j:n]
    end
end

struct CholeskyVec
    n::Int
    uplo::Char
end
function (c::CholeskyVec)(x::LA.Cholesky)
    cartesian_indices = _get_cartesian_indices(c.n, c.uplo)
    return [x.UL[i, j] for (i, j) in cartesian_indices]
end
function with_logabsdet_jacobian(c::CholeskyVec, x::LA.Cholesky{T}) where {T<:Number}
    return (c(x), zero(T))
end

struct CholeskyUnVec
    n::Int
    uplo::Char
end
function (c::CholeskyUnVec)(xvec::AbstractVector{T}) where {T<:Number}
    x = if c.uplo == 'U'
        LA.Cholesky(LA.UpperTriangular(zeros(T, c.n, c.n)))
    else
        LA.Cholesky(LA.LowerTriangular(zeros(T, c.n, c.n)))
    end
    cartesian_indices = _get_cartesian_indices(c.n, c.uplo)
    for (idx, (i, j)) in enumerate(cartesian_indices)
        x.UL[i, j] = xvec[idx]
    end
    return x
end
function with_logabsdet_jacobian(c::CholeskyUnVec, x::AbstractVector{T}) where {T<:Number}
    return (c(x), zero(T))
end
function optic_vec(d::D.LKJCholesky)
    n = first(size(d))
    sym = if d.uplo == 'U'
        :U
    else
        :L
    end
    return [
        AbstractPPL.@opticof(_.$sym[i, j]) for (i, j) in _get_cartesian_indices(n, d.uplo)
    ]
end

from_vec(d::D.LKJCholesky) = CholeskyUnVec(first(size(d)), d.uplo)
to_vec(d::D.LKJCholesky) = CholeskyVec(first(size(d)), d.uplo)
function vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
from_linked_vec(d::D.LKJCholesky) = inverse(B.VecCholeskyBijector(d.uplo))
to_linked_vec(d::D.LKJCholesky) = B.VecCholeskyBijector(d.uplo)
function linked_vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
function linked_optic_vec(d::D.LKJCholesky)
    return fill(nothing, linked_vec_length(d))
end
