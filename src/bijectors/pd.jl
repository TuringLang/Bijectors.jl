struct PDBijector <: Bijector{2} end

# This function has custom adjoints defined for Tracker, Zygote and ReverseDiff.
# I couldn't find a mutation-free implementation that maintains TrackedArrays in Tracker
# and ReverseDiff, hence the need for custom adjoints.
function replace_diag(f, X)
    g(i, j) = ifelse(i == j, f(X[i, i]), X[i, j])
    return g.(1:size(X, 1), (1:size(X, 2))')
end
function (b::PDBijector)(X::AbstractMatrix{<:Real})
    Y = cholesky(X; check = true).L
    return replace_diag(log, Y)
end
(b::PDBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

function (ib::Inverse{<:PDBijector})(Y::AbstractMatrix{<:Real})
    X = replace_diag(exp, Y)
    return LowerTriangular(X) * LowerTriangular(X)'
end
(ib::Inverse{<:PDBijector})(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, X)

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
    return - sum((d .- (1:d) .+ 2) .* log.(diag(U))) + d * log(T(2))
end

logabsdetjac(b::PDBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end
