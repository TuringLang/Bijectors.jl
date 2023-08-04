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

cholesky_factor(X::AbstractMatrix) = cholesky_factor(cholesky(Hermitian(X)))
cholesky_factor(X::Cholesky) = X.U
cholesky_factor(X::UpperTriangular) = X
cholesky_factor(X::LowerTriangular) = X

# TODO: Add `check` as an argument?
"""
    cholesky_lower(X)

Return the lower triangular Cholesky factor of `X` as a `Matrix`
rather than `LowerTriangular`.

!!! note
    This is a thin wrapper around `cholesky(Hermitian(X)).L`
    but with a custom `ChainRulesCore.rrule` implementation.
"""
cholesky_lower(X::AbstractMatrix) = lower_triangular(parent(cholesky(Hermitian(X)).L))
function ChainRulesCore.rrule(::typeof(cholesky_lower), X::AbstractMatrix)
    project_to = ChainRulesCore.ProjectTo(X)
    # TODO: Do we need a special rule for `Hermitian`?
    H, hermitian_pullback = ChainRulesCore.rrule(Hermitian, X, :L)
    C, cholesky_pullback = ChainRulesCore.rrule(cholesky, H, Val(false))
    function cholesky_lower_pullback(_ΔL)
        ΔL = ChainRulesCore.unthunk(_ΔL)
        ΔC = ChainRulesCore.Tangent{typeof(C)}(; factors=(C.uplo === :L ? ΔL : ΔL'))
        ΔH = cholesky_pullback(ΔC)[2]
        Δx = hermitian_pullback(ΔH)[2]
        # No need to add pullback for `lower_triangular`, because the pullback
        # for `Hermitian` already produces the correct result (i.e. the lower-triangular
        # part zeroed out).
        return (ChainRulesCore.NoTangent(), project_to(Δx))
    end

    return lower_triangular(parent(C.L)), cholesky_lower_pullback
end

"""
    cholesky_upper(X)

Return the upper triangular Cholesky factor of `X` as a `Matrix`
rather than `UpperTriangular`.

!!! note
    This is a thin wrapper around `cholesky(Hermitian(X)).U`
    but with a custom `ChainRulesCore.rrule` implementation.
"""
cholesky_upper(X::AbstractMatrix) = upper_triangular(parent(cholesky(Hermitian(X)).U))
function ChainRulesCore.rrule(::typeof(cholesky_upper), X::AbstractMatrix)
    H, hermitian_pullback = ChainRulesCore.rrule(Hermitian, X, :U)
    C, cholesky_pullback = ChainRulesCore.rrule(cholesky, H, Val(false))
    function cholesky_upper_pullback(_ΔU)
        ΔU = ChainRulesCore.unthunk(_ΔU)
        ΔC = ChainRulesCore.Tangent{typeof(C)}(; factors=(C.uplo === :U ? ΔU : ΔU'))
        ΔH = cholesky_pullback(ΔC)[2]
        Δx = hermitian_pullback(ΔH)[2]
        # No need to add pullback for `upper_triangular`, because the pullback
        # for `Hermitian` already produces the correct result (i.e. the upper-triangular
        # part zeroed out).
        return (ChainRulesCore.NoTangent(), Δx)
    end

    return upper_triangular(parent(C.U)), cholesky_upper_pullback
end

"""
    triu_mask(X::AbstractMatrix, k::Int)

Return a mask for elements of `X` above the `k`th diagonal.
"""
function triu_mask(X::AbstractMatrix, k::Int)
    # Ensure that we're working with a square matrix.
    LinearAlgebra.checksquare(X)

    # Using `similar` allows us to respect device of array, etc., e.g. `CuArray`.
    m = similar(X, Bool)
    return triu!(fill!(parent(m), true), k)
end

ChainRulesCore.@non_differentiable triu_mask(X::AbstractMatrix, k::Int)

_triu_to_vec(X::AbstractMatrix{<:Real}, k::Int) = X[triu_mask(X, k)]

function update_triu_from_vec!(
    vals::AbstractVector{<:Real}, k::Int, X::AbstractMatrix{<:Real}
)
    # Ensure that we're working with one-based indexing.
    # `triu` requires this too.
    LinearAlgebra.require_one_based_indexing(X)

    # Set the values.
    idx = 1
    m, n = size(X)
    for j in 1:n
        for i in 1:min(j - k, m)
            X[i, j] = vals[idx]
            idx += 1
        end
    end

    return X
end

function update_triu_from_vec(vals::AbstractVector{<:Real}, k::Int, dim::Int)
    X = similar(vals, dim, dim)
    # TODO: Do we need this?
    fill!(X, 0)
    return update_triu_from_vec!(vals, k, X)
end

function ChainRulesCore.rrule(
    ::typeof(update_triu_from_vec), x::AbstractVector{<:Real}, k::Int, dim::Int
)
    function update_triu_from_vec_pullback(ΔX)
        return (
            ChainRulesCore.NoTangent(),
            _triu_to_vec(ChainRulesCore.unthunk(ΔX), k),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end
    return update_triu_from_vec(x, k, dim), update_triu_from_vec_pullback
end

#      n * (n - 1) / 2 = d
# ⟺       n^2 - n - 2d = 0
# ⟹                  n = (1 + sqrt(1 + 8d)) / 2
_triu1_dim_from_length(d) = (1 + isqrt(1 + 8d)) ÷ 2

"""
    triu1_to_vec(X::AbstractMatrix{<:Real})

Extracts elements from upper triangle of `X` with offset `1` and returns them as a vector.
"""
triu1_to_vec(X::AbstractMatrix) = _triu_to_vec(X, 1)

inverse(::typeof(triu1_to_vec)) = vec_to_triu1

"""
    vec_to_triu1(x::AbstractVector{<:Real})

Constructs a matrix from a vector `x` by filling the upper triangle with offset `1`.
"""
function vec_to_triu1(x::AbstractVector)
    n = _triu1_dim_from_length(length(x))
    X = update_triu_from_vec(x, 1, n)
    return upper_triangular(X)
end

inverse(::typeof(vec_to_triu1)) = triu1_to_vec

function vec_to_triu1_row_index(idx)
    # Assumes that vector was saved in a column-major order
    # and that vector is one-based indexed.
    M = _triu1_dim_from_length(idx - 1)
    return idx - (M * (M - 1) ÷ 2)
end

# Triangular matrix with diagonals.

#     (n^2 + n) / 2 = d
# ⟺    n² + n - 2d = 0
# ⟺              n = (-1 + sqrt(1 + 8d)) / 2
_triu_dim_from_length(d) = (-1 + isqrt(1 + 8 * d)) ÷ 2

"""
    triu_to_vec(X::AbstractMatrix{<:Real})

Extracts elements from upper triangle of `X` and returns them as a vector.
"""
triu_to_vec(X::AbstractMatrix) = _triu_to_vec(X, 0)

"""
    vec_to_triu(x::AbstractVector{<:Real})

Constructs a matrix from a vector `x` by filling the upper triangle.
"""
function vec_to_triu(x::AbstractVector)
    n = _triu_dim_from_length(length(x))
    X = update_triu_from_vec(x, 0, n)
    return upper_triangular(X)
end

inverse(::typeof(vec_to_triu)) = triu_to_vec
