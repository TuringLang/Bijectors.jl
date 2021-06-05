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
    Y = lower(parent(cholesky(X; check = true).L))
    return replace_diag(log, Y)
end
lower(A::AbstractMatrix) = convert(typeof(A), LowerTriangular(A))

function transform(ib::Inverse{<:PDBijector}, Y::AbstractMatrix{<:Real})
    X = replace_diag(exp, Y)
    return getpd(X)
end
getpd(X) = LowerTriangular(X) * LowerTriangular(X)'

function logabsdetjac(b::PDBijector, X::AbstractMatrix{<:Real})
    T = eltype(X)
    Xcf = cholesky(X, check = false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I)
    end
    return logabsdetjac(b, Xcf)
end

function logabsdetjac(b::PDBijector, Xcf::Cholesky)
    U = Xcf.U
    T = eltype(U)
    d = size(U, 1)
    return - sum((d .- (1:d) .+ 2) .* log.(diag(U))) - d * log(T(2))
end
