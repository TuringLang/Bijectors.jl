using SparseArrays

struct Permute{A} <: Bijector{1}
    A::A
end

function Permute(indices::AbstractVector{Int})
    # construct a sparse-matrix for use in the multiplication
    n = length(indices)
    A = spzeros(n, n)

    for (i, idx) in enumerate(indices)
        A[idx, i] = 1.0
    end

    return Permute(A)
end

@inline (b::Permute)(x::AbstractVector) = b.A * x
@inline inv(b::Permute) = Permute(transpose(b.A))

logabsdetjac(b, x::AbstractVector) = zero(eltype(x))
logabsdetjac(b, x::AbstractMatrix) = zero(eltype(x), size(x, 2))
