struct PDBijector <: Bijector end

# This function has custom adjoints defined for Tracker, Zygote and ReverseDiff.
# I couldn't find a mutation-free implementation that maintains TrackedArrays in Tracker
# and ReverseDiff, hence the need for custom adjoints.
function replace_diag(f, X)
    g(i, j) = ifelse(i == j, f(X[i, i]), X[i, j])
    return g.(1:size(X, 1), (1:size(X, 2))')
end
transform(b::PDBijector, X::AbstractMatrix{<:Real}) = pd_link(X)
function pd_link(X)
    Y = lower_triangular(parent(cholesky(X; check=true).L))
    return replace_diag(log, Y)
end

function transform(ib::Inverse{PDBijector}, Y::AbstractMatrix{<:Real})
    X = replace_diag(exp, Y)
    return pd_from_lower(X)
end

function logabsdetjac(b::PDBijector, X::AbstractMatrix{<:Real})
    T = eltype(X)
    Xcf = cholesky(X; check=false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I)
    end
    return logabsdetjac_pdbijector_chol(Xcf)
end

function logabsdetjac_pdbijector_chol(Xcf::Cholesky)
    # NOTE: Use `UpperTriangular` here because we only need `diag(U)`
    # and `UL` is by default already constructed in `Cholesky`.
    UL = Xcf.UL
    d = size(UL, 1)
    z = sum(((d + 1):(-1):2) .* log.(diag(UL)))
    return -(z + d * oftype(z, IrrationalConstants.logtwo))
end

# TODO: Implement explicitly.
function with_logabsdet_jacobian(b::PDBijector, X)
    return transform(b, X), logabsdetjac(b, X)
end


struct PDVecBijector <: Bijector end

function _triu_dim_from_length(d)
    #           d = n^2 + n
    # n² + n - 2d = 0
    #           n = (-1 + sqrt(1 + 8d)) / 2
    return (-1 + isqrt(1 + 8 * d)) ÷ 2
end

"""
    vec_to_triu(x::AbstractVector{<:Real})

Constructs a matrix from a vector `x` by filling the upper triangle.
"""
function vec_to_triu(x::AbstractVector)
    n = _triu_dim_from_length(length(x))
    X = update_triu_from_vec(x, 0, n)
    return upper_triangular(X)
end

transform(::PDVecBijector, X::AbstractMatrix{<:Real}) = pd_vec_link(X)
pd_vec_link(X) = triu_to_vec(transpose(pd_link(X)), 0)

function transform(::Inverse{PDVecBijector}, y::AbstractVector{<:Real})
    Y = permutedims(vec_to_triu(y))
    return transform(inverse(PDBijector()), Y)
end

logabsdetjac(::PDVecBijector, X::AbstractMatrix{<:Real}) = logabsdetjac(PDBijector(), X)
