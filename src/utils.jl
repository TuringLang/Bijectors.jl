# `permutedims` seems to work better with AD (cf. KernelFunctions.jl)
aT_b(a::AbstractVector{<:Real}, b::AbstractMatrix{<:Real}) = permutedims(a) * b
# `permutedims` can't be used here since scalar output is desired
aT_b(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = dot(a, b)

# flatten arrays with fallback for scalars
_vec(x::AbstractArray{<:Real}) = vec(x)
_vec(x::Real) = x

# # Because `ReverseDiff` does not play well with structural matrices.
lower_triangular(A::AbstractMatrix) = convert(typeof(A), LowerTriangular(A))
upper_triangular(A::AbstractMatrix) = convert(typeof(A), UpperTriangular(A))

pd_from_lower(X) = LowerTriangular(X) * LowerTriangular(X)'
pd_from_upper(X) = UpperTriangular(X)' * UpperTriangular(X)

cholesky_factor(X::AbstractMatrix) = cholesky(X).UL
cholesky_factor(X::Cholesky) = X.UL
cholesky_factor(X::UpperTriangular) = X
cholesky_factor(X::LowerTriangular) = X
