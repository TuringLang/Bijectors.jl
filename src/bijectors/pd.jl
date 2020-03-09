struct PDBijector <: Bijector{2} end

function replace_diag(X, y)
    f(i, j) = ifelse(i == j, y[i], X[i, j])
    return f.(1:size(X, 1), (1:size(X, 2))')
end
function (b::PDBijector)(X::AbstractMatrix{<:Real})
    Y = cholesky(X).L
    return replace_diag(Y, log.(diag(Y)))
end
function (ib::Inverse{<:PDBijector})(Y::AbstractMatrix{<:Real})
    X = replace_diag(Y, exp.(diag(Y)))
    return LowerTriangular(X) * LowerTriangular(X)'
end
