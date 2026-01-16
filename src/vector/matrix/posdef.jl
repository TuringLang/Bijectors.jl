# Distributions over positive (semi)definite matrices.
#
# We could in principle just use PDVecBijector, which does the same thing as this
# reimplemented version. However, (1) PDVecBijector has a slightly convoluted definition,
# and this one is probably faster; (2) ReverseDiff chokes on PDVecBijector:
# https://github.com/TuringLang/Bijectors.jl/issues/432

import LinearAlgebra as LA
import IrrationalConstants: logtwo

const PDMatrixDistribution = Union{D.MatrixBeta,D.Wishart,D.InverseWishart}

struct PosDef
    original_size::Int
end
function (p::PosDef)(x::AbstractMatrix{T}) where {T<:Real}
    # This is technically inefficient as it performs a few extra multiplications
    # and additions.
    return first(with_logabsdet_jacobian(p, x))
end
function with_logabsdet_jacobian(p::PosDef, x::AbstractMatrix{T}) where {T<:Real}
    d = p.original_size
    yvec = zeros(T, div(d * (d + 1), 2))
    L = LA.cholesky(LA.Hermitian(x, :L)).L
    idx = 1
    z = zero(T)
    weight = d + 1
    for i in 1:d
        for j in 1:i
            if i == j
                logLij = log(L[i, j])
                yvec[idx] = logLij
                z -= weight * logLij
                weight -= 1
            else
                yvec[idx] = L[i, j]
            end
            idx += 1
        end
    end
    logjac = z - (d * oftype(z, logtwo))
    return yvec, logjac
end
inverse(p::PosDef) = InvPosDef(p.original_size)

struct InvPosDef
    original_size::Int
end
function (ip::InvPosDef)(yvec::AbstractVector{T}) where {T<:Real}
    # Like above, this is technically inefficient as it performs a few extra multiplications
    # and additions.
    return first(with_logabsdet_jacobian(ip, yvec))
end
function with_logabsdet_jacobian(ip::InvPosDef, yvec::AbstractVector{T}) where {T<:Real}
    d = ip.original_size
    X = zeros(T, d, d)
    idx = 1
    z = zero(T)
    weight = d + 1
    for i in 1:d
        for j in 1:i
            if i == j
                X[i, j] = exp(yvec[idx])
                z += weight * yvec[idx]
                weight -= 1
            else
                X[i, j] = yvec[idx]
            end
            idx += 1
        end
    end
    logjac = z + (d * oftype(z, logtwo))
    return X * X', logjac
end
inverse(ip::InvPosDef) = PosDef(ip.original_size)

from_linked_vec(d::PDMatrixDistribution) = InvPosDef(first(size(d)))
to_linked_vec(d::PDMatrixDistribution) = PosDef(first(size(d)))
function linked_vec_length(d::PDMatrixDistribution)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
linked_optic_vec(d::PDMatrixDistribution) = fill(nothing, linked_vec_length(d))
