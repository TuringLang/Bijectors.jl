struct PDBijector <: Bijector{2} end
function (b::PDBijector)(X::AbstractMatrix{<:Real})
    Y = Matrix(cholesky(X).L)
    f(x, i, j) = i == j ? log(x) : x
    Y = f.(Y, 1:size(Y,1), (1:size(Y,2))')
    return Y
end
function (ib::Inverse{<:PDBijector})(Y::AbstractMatrix{<:Real})
    f(x, i, j) = i == j ? exp(x) : x
    X = f.(Y, 1:size(Y,1), (1:size(Y,2))')
    return LowerTriangular(X) * LowerTriangular(X)'
end
