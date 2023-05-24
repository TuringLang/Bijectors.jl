"""
    CorrBijector <: Bijector

A bijector implementation of Stan's parametrization method for Correlation matrix:
https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html

Basically, a unconstrained strictly upper triangular matrix `y` is transformed to 
a correlation matrix by following readable but not that efficient form:

```
K = size(y, 1)
z = tanh.(y)

for j=1:K, i=1:K
    if i>j
        w[i,j] = 0
    elseif 1==i==j
        w[i,j] = 1
    elseif 1<i==j
        w[i,j] = prod(sqrt(1 .- z[1:i-1, j].^2))
    elseif 1==i<j
        w[i,j] = z[i,j]
    elseif 1<i<j
        w[i,j] = z[i,j] * prod(sqrt(1 .- z[1:i-1, j].^2))
    end
end
```

It is easy to see that every column is a unit vector, for example:

```
w3' w3 ==
w[1,3]^2 + w[2,3]^2 + w[3,3]^2 ==
z[1,3]^2 + (z[2,3] * sqrt(1 - z[1,3]^2))^2 + (sqrt(1-z[1,3]^2) * sqrt(1-z[2,3]^2))^2 ==
z[1,3]^2 + z[2,3]^2 * (1-z[1,3]^2) + (1-z[1,3]^2) * (1-z[2,3]^2) ==
z[1,3]^2 + z[2,3]^2 - z[2,3]^2 * z[1,3]^2 + 1 -z[1,3]^2 - z[2,3]^2 + z[1,3]^2 * z[2,3]^2 ==
1
```

And diagonal elements are positive, so `w` is a cholesky factor for a positive matrix.

```
x = w' * w
```

Consider block matrix representation for `x`

```
x = [w1'; w2'; ... wn'] * [w1 w2 ... wn] == 
[w1'w1 w1'w2 ... w1'wn;
 w2'w1 w2'w2 ... w2'wn;
 ...
]
```

The diagonal elements are given by `wk'wk = 1`, thus `x` is a correlation matrix.

Every step is invertible, so this is a bijection(bijector).

Note: The implementation doesn't follow their "manageable expression" directly,
because their equation seems wrong (7/30/2020). Insteadly it follows definition 
above the "manageable expression" directly, which is also described in above doc.
"""
struct CorrBijector <: Bijector end

with_logabsdet_jacobian(b::CorrBijector, x) = transform(b, x), logabsdetjac(b, x)

function transform(b::CorrBijector, X::AbstractMatrix{<:Real})
    w = upper_triangular(parent(cholesky(X).U))  # keep LowerTriangular until here can avoid some computation
    r = _link_chol_lkj(w) 
    return r + zero(X) 
    # This dense format itself is required by a test, though I can't get the point.
    # https://github.com/TuringLang/Bijectors.jl/blob/b0aaa98f90958a167a0b86c8e8eca9b95502c42d/test/transform.jl#L67
end

function transform(ib::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})
    w = _inv_link_chol_lkj(y)
    return pd_from_upper(w)
end

logabsdetjac(::Inverse{CorrBijector}, Y::AbstractMatrix{<:Real}) = _logabsdetjac_inv_corr(Y)
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    #=
    It may be more efficient if we can use un-contraint value to prevent call of b
    It's recommended to directly call 
    `logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})`
    if possible.
    =#
    return -logabsdetjac(inverse(b), (b(X)))
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
    return triu(.~m .| m, k)
end

triu_to_vec(X::AbstractMatrix{<:Real}, k::Int) = X[triu_mask(X, k)]

function update_triu_from_vec!(
    vals::AbstractVector{<:Real},
    k::Int,
    X::AbstractMatrix{<:Real}
)
    # Ensure that we're working with one-based indexing.
    # `triu` requires this too.
    LinearAlgebra.require_one_based_indexing(X)

    # Set the values.
    idx = 1
    m, n = size(X)
    for j = 1:n
        for i = 1:min(j - k, m)
            X[i, j] = vals[idx]
            idx += 1
        end
    end

    return X
end

function update_triu_from_vec(vals::AbstractVector{<:Real}, k::Int, dim::Int)
    X = similar(vals, dim, dim)
    # TODO: Do we need this?
    X .= 0
    return update_triu_from_vec!(vals, k, X)
end

function ChainRulesCore.rrule(::typeof(update_triu_from_vec), x::AbstractVector{<:Real}, k::Int, dim::Int)
    function update_triu_from_vec_pullback(ΔX)
        return (
            ChainRulesCore.NoTangent(),
            triu_to_vec(ChainRulesCore.unthunk(ΔX), k),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent()
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
triu1_to_vec(X::AbstractMatrix) = triu_to_vec(X, 1)

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
    return idx - (M*(M-1) ÷ 2)
end

"""
    VecCorrBijector <: Bijector

A bijector to transform a correlation matrix to an unconstrained vector. 

# Reference
https://mc-stan.org/docs/reference-manual/correlation-matrix-transform.html

See also: [`CorrBijector`](@ref) and ['VecCholeskyBijector'](@ref)

# Example

```jldoctest
julia> using LinearAlgebra

julia> using StableRNGs; rng = StableRNG(42);

julia> b = Bijectors.VecCorrBijector();

julia> X = rand(rng, LKJ(3, 1))  # Sample a correlation matrix.
3×3 Matrix{Float64}:
  1.0       -0.705273   -0.348638
 -0.705273   1.0         0.0534538
 -0.348638   0.0534538   1.0

julia> y = b(X)  # Transform to unconstrained vector representation.
3-element Vector{Float64}:
 -0.8777149781928181
 -0.3638927608636788
 -0.29813769428942216

julia> inverse(b)(y) ≈ X  # (✓) Round-trip through `b` and its inverse.
true
"""
struct VecCorrBijector <: Bijector end

with_logabsdet_jacobian(b::VecCorrBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(::VecCorrBijector, X) = _link_chol_lkj(cholesky_factor(X))

function logabsdetjac(b::VecCorrBijector, x)
    return -logabsdetjac(inverse(b), b(x))
end

transform(::Inverse{VecCorrBijector}, y::AbstractVector{<:Real}) = pd_from_upper(_inv_link_chol_lkj(y))

logabsdetjac(::Inverse{VecCorrBijector}, y::AbstractVector{<:Real}) = _logabsdetjac_inv_corr(y)

"""
    VecCholeskyBijector <: Bijector

A bijector to transform a Cholesky factor of a correlation matrix to an unconstrained vector. 

# Fields
- mode :`Symbol`. Controls the inverse tranformation :
    - if `mode === :U` returns a `LinearAlgebra.Cholesky` holding the `UpperTriangular` factor
    - if `mode === :L` returns a `LinearAlgebra.Cholesky` holding the `LowerTriangular` factor

# Reference
https://mc-stan.org/docs/reference-manual/cholesky-factors-of-correlation-matrices-1

See also: [`VecCorrBijector`](@ref)

# Example

```jldoctest
julia> using LinearAlgebra

julia> using StableRNGs; rng = StableRNG(42);

julia> b = Bijectors.VecCholeskyBijector(:U);

julia> X = rand(rng, LKJCholesky(3, 1, :U))  # Sample a correlation matrix.
Cholesky{Float64, Matrix{Float64}}
U factor:
3×3 UpperTriangular{Float64, Matrix{Float64}}:
 1.0  0.937494   0.865891
  ⋅   0.348002  -0.320442
  ⋅    ⋅         0.384122

julia> y = b(X)  # Transform to unconstrained vector representation.
3-element Vector{Float64}:
 -0.8777149781928181
 -0.3638927608636788
 -0.29813769428942216

julia> X_inv = inverse(b)(y); 
julia> X_inv.U ≈ X.U  # (✓) Round-trip through `b` and its inverse.
true
julia> X_inv.L ≈ X.L  # (✓) Also works for the lower triangular factor.
true
"""
struct VecCholeskyBijector <: Bijector 
    mode::Symbol
    function VecCholeskyBijector(uplo)
        s = Symbol(uplo)
        if (s === :U) || (s === :L)
            new(s)
        else
            throw(ArgumentError("mode must be either :U (upper triangular) or :L (lower triangular)"))
        end
    end
end

# TODO: Implement directly to make use of shared computations.
with_logabsdet_jacobian(b::VecCholeskyBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(::VecCholeskyBijector, X) = _link_chol_lkj(cholesky_factor(X))

function logabsdetjac(b::VecCholeskyBijector, x)
    return -logabsdetjac(inverse(b), b(x))
end

function transform(b::Inverse{VecCholeskyBijector}, y::AbstractVector{<:Real})
    if b.orig.mode === :U
        # This Cholesky constructor is compatible with Julia v1.6
        # for later versions Cholesky(::UpperTriangular) works
        return Cholesky(_inv_link_chol_lkj(y), 'U', 0)
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        # HACK: Need to make materialize the transposed matrix to avoid numerical instabilities.
        # If we don't, the return-type can be both `Matrix` and `Transposed`.
        return Cholesky(Matrix(transpose(_inv_link_chol_lkj(y))), 'L', 0)
    end
end

logabsdetjac(::Inverse{VecCholeskyBijector}, y::AbstractVector{<:Real}) = _logabsdetjac_inv_chol(y)

"""
    function _link_chol_lkj(w)

Link function for cholesky factor.

An alternative and maybe more efficient implementation was considered:

```
for i=2:K, j=(i+1):K
    z[i, j] = (w[i, j] / w[i-1, j]) * (z[i-1, j] / sqrt(1 - z[i-1, j]^2))
end
```

But this implementation will not work when w[i-1, j] = 0.
Though it is a zero measure set, unit matrix initialization will not work.

For equivalence, following explanations is given by @torfjelde:

For `(i, j)` in the loop below, we define

    z₍ᵢ₋₁, ⱼ₎ = w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² (1 / √(1 - z₍ₖ,ⱼ₎²))

and so

    z₍ᵢ,ⱼ₎ = w₍ᵢ,ⱼ₎ * ∏ₖ₌₁ⁱ⁻¹ (1 / √(1 - z₍ₖ,ⱼ₎²))
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²))
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²)) / w₍ᵢ₋₁,ⱼ₎
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (z₍ᵢ₋₁,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎)
            = (w₍ᵢ,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎) * (z₍ᵢ₋₁,ⱼ₎ / √(1 - z₍ᵢ₋₁,ⱼ₎²))

which is the above implementation.
"""
function _link_chol_lkj(W::AbstractMatrix)
    # TODO: Implement adjoint to support reverse-mode AD backends properly.
    K = LinearAlgebra.checksquare(W)

    z = similar(W) # z is also UpperTriangular. 
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    # This block can't be integrated with loop below, because W[1,1] != 0.
    @inbounds z[1, 1] = 0

    @inbounds for j = 2:K
        z[1, j] = atanh(W[1, j])
        tmp = sqrt(1 - W[1, j]^2)
        for i in 2:(j-1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            z[i, j] = atanh(p)
        end
        z[j, j] = 0
    end

    return z
end

function _link_chol_lkj(W::UpperTriangular)
    K = LinearAlgebra.checksquare(W)
    N = ((K-1)*K) ÷ 2   # {K \choose 2} free parameters

    z = similar(W, N)

    idx = 1
    @inbounds for j = 2:K
        z[idx] = atanh(W[1, j])
        idx += 1
        tmp = sqrt(1 - W[1, j]^2)
        for i in 2:(j-1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            z[idx] = atanh(p)
            idx += 1
        end
    end

    return z
end

_link_chol_lkj(W::LowerTriangular) = _link_chol_lkj(transpose(W))

"""
    _inv_link_chol_lkj(y)

Inverse link function for cholesky factor.
"""
function _inv_link_chol_lkj(Y::AbstractMatrix)
    # TODO: Implement adjoint to support reverse-mode AD backends properly.
    K = LinearAlgebra.checksquare(Y)

    W = similar(Y)

    @inbounds for j in 1:K
        W[1, j] = 1
        for i in 2:j
            z = tanh(Y[i-1, j])
            tmp = W[i-1, j]
            W[i-1, j] = z * tmp
            W[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j+1):K
            W[i, j] = 0
        end
    end

    return W
end

function _inv_link_chol_lkj(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    W = similar(y, K, K)

    idx = 1
    @inbounds for j in 1:K
        W[1, j] = 1
        for i in 2:j
            z = tanh(y[idx])
            idx += 1
            tmp = W[i-1, j]
            W[i-1, j] = z * tmp
            W[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j+1):K
            W[i, j] = 0
        end
    end

    return W
end

function _logabsdetjac_inv_corr(Y::AbstractMatrix)
    K = LinearAlgebra.checksquare(Y)

    result = float(zero(eltype(Y)))
    for j in 2:K, i in 1:(j-1)
        @inbounds abs_y_i_j = abs(Y[i, j])
        result += (K - i + 1) * (
            IrrationalConstants.logtwo - (abs_y_i_j + LogExpFunctions.log1pexp(-2 * abs_y_i_j))
        )
    end
    return result
end

function _logabsdetjac_inv_corr(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    for (i, y_i) in enumerate(y)
        abs_y_i = abs(y_i)
        row_idx = vec_to_triu1_row_index(i)
        result += (K - row_idx + 1) * (
            IrrationalConstants.logtwo - (abs_y_i + LogExpFunctions.log1pexp(-2 * abs_y_i))
        )
    end
    return result
end

function _logabsdetjac_inv_chol(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    idx = 1
    @inbounds for j in 2:K
        tmp = zero(result)
        for _ in 1:(j-1)
            z = tanh(y[idx])
            logz = log(1 - z^2)
            result += logz + (tmp / 2)
            tmp += logz
            idx += 1
        end
    end

    return result
end
