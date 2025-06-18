struct PDBijector <: Bijector end

# This function has custom adjoints defined for Tracker and ReverseDiff.
# I couldn't find a mutation-free implementation that maintains TrackedArrays in Tracker
# and ReverseDiff, hence the need for custom adjoints.
function replace_diag(f, X)
    g(i, j) = ifelse(i == j, f(X[i, i]), X[i, j])
    return g.(1:size(X, 1), (1:size(X, 2))')
end
transform(b::PDBijector, X::AbstractMatrix{<:Real}) = pd_link(X)
pd_link(X) = replace_diag(log, cholesky_lower(X))

function transform(ib::Inverse{PDBijector}, Y::AbstractMatrix{<:Real})
    X = replace_diag(exp, Y)
    return pd_from_lower(X)
end

function logabsdetjac(b::PDBijector, X::AbstractMatrix{<:Real})
    L = cholesky_lower(X)
    return logabsdetjac_pdbijector_chol(L)
end

function logabsdetjac_pdbijector_chol(X::AbstractMatrix)
    d = size(X, 1)
    z = sum(((d + 1):(-1):2) .* log.(diag(X)))
    return -(z + d * oftype(z, IrrationalConstants.logtwo))
end

function with_logabsdet_jacobian(b::PDBijector, X)
    L = cholesky_lower(X)
    return replace_diag(log, L), logabsdetjac_pdbijector_chol(L)
end

struct PDVecBijector <: Bijector end

transform(::PDVecBijector, X::AbstractMatrix{<:Real}) = pd_vec_link(X)
# TODO: Implement `tril_to_vec` and remove `permutedims`.
pd_vec_link(X) = triu_to_vec(transpose_eager(pd_link(X)))

function transform(::Inverse{PDVecBijector}, y::AbstractVector{<:Real})
    Y = transpose_eager(vec_to_triu(y))
    return transform(inverse(PDBijector()), Y)
end

logabsdetjac(::PDVecBijector, X::AbstractMatrix{<:Real}) = logabsdetjac(PDBijector(), X)

function with_logabsdet_jacobian(b::PDVecBijector, X)
    return transform(b, X), logabsdetjac(b, X)
end

function output_size(::PDVecBijector, sz::Tuple{Int,Int})
    n = first(sz)
    d = (n^2 + n) ÷ 2
    return (d,)
end

function output_size(::Inverse{PDVecBijector}, sz::Tuple{Int})
    n = _triu_dim_from_length(first(sz))
    return (n, n)
end
