"""
    CorrBijector <: Bijector

A bijector implementation of Stan's parametrization method for Correlation matrix:
https://mc-stan.org/docs/reference-manual/transforms.html#correlation-matrix-transform.section

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

function transform(b::CorrBijector, X)
    w = cholesky_upper(X)
    r = _link_chol_lkj(w)
    return r
end

function with_logabsdet_jacobian(ib::Inverse{CorrBijector}, y)
    U, logJ = _inv_link_chol_lkj(y)
    K = size(U, 1)
    for j in 2:(K - 1)
        logJ += (K - j) * log(U[j, j])
    end
    return pd_from_upper(U), logJ
end

logabsdetjac(::Inverse{CorrBijector}, Y) = _logabsdetjac_inv_corr(Y)
function logabsdetjac(b::CorrBijector, X)
    #=
    It may be more efficient if we can use un-contraint value to prevent call of b
    It's recommended to directly call 
    `logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})`
    if possible.
    =#
    return -logabsdetjac(inverse(b), (b(X)))
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

transform(::VecCorrBijector, X) = _link_chol_lkj_from_upper(cholesky_upper(X))

function logabsdetjac(b::VecCorrBijector, x)
    return -logabsdetjac(inverse(b), b(x))
end

function with_logabsdet_jacobian(::Inverse{VecCorrBijector}, y)
    U_logJ = _inv_link_chol_lkj(y)
    # workaround for `Tracker.TrackedTuple` not supporting iteration
    U, logJ = U_logJ[1], U_logJ[2]
    K = size(U, 1)
    for j in 2:(K - 1)
        logJ += (K - j) * log(U[j, j])
    end
    return pd_from_upper(U), logJ
end

function logabsdetjac(::Inverse{VecCorrBijector}, y)
    return _logabsdetjac_inv_corr(y)
end

function output_size(::VecCorrBijector, sz::Tuple{Int,Int})
    sz[1] == sz[2] || error("sizes should be equal; received $(sz)")
    n = sz[1]
    return ((n * (n - 1)) ÷ 2,)
end

function output_size(::Inverse{VecCorrBijector}, sz::Tuple{Int})
    n = _triu1_dim_from_length(first(sz))
    return (n, n)
end

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
            throw(
                ArgumentError(
                    "mode must be either :U (upper triangular) or :L (lower triangular)"
                ),
            )
        end
    end
end

# TODO: Implement directly to make use of shared computations.
with_logabsdet_jacobian(b::VecCholeskyBijector, x) = transform(b, x), logabsdetjac(b, x)

function transform(b::VecCholeskyBijector, X)
    return if b.mode === :U
        _link_chol_lkj_from_upper(cholesky_upper(X))
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        _link_chol_lkj_from_lower(cholesky_lower(X))
    end
end

function logabsdetjac(b::VecCholeskyBijector, x)
    return -logabsdetjac(inverse(b), b(x))
end

function with_logabsdet_jacobian(b::Inverse{VecCholeskyBijector}, y::AbstractVector{<:Real})
    factors, logJ = _inv_link_chol_lkj(y)
    if b.orig.mode === :U
        # This Cholesky constructor is compatible with Julia v1.6
        # for later versions Cholesky(::UpperTriangular) works
        return Cholesky(factors, 'U', 0), logJ
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        # HACK: Need to make materialize the transposed matrix to avoid numerical instabilities.
        # If we don't, the return-type can be both `Matrix` and `Transposed`.
        return Cholesky(transpose_eager(factors), 'L', 0), logJ
    end
end

function logabsdetjac(::Inverse{VecCholeskyBijector}, y::AbstractVector{<:Real})
    return _logabsdetjac_inv_chol(y)
end

output_size(::VecCholeskyBijector, sz::Tuple{Int,Int}) = output_size(VecCorrBijector(), sz)
function output_size(::Inverse{<:VecCholeskyBijector}, sz::Tuple{Int})
    return output_size(inverse(VecCorrBijector()), sz)
end

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
    K = LinearAlgebra.checksquare(W)

    y = similar(W) # W is upper triangular.
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    @inbounds for j in 1:K
        remainder_sq = W[j, j]^2
        for i in (j - 1):-1:1
            z = W[i, j] / sqrt(remainder_sq)
            y[i, j] = asinh(z)
            remainder_sq += W[i, j]^2
        end
        for i in j:K
            y[i, j] = 0
        end
    end

    return y
end

function _link_chol_lkj_from_upper(W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)
    N = ((K - 1) * K) ÷ 2   # {K \choose 2} free parameters

    y = similar(W, N)

    starting_idx = 1
    @inbounds for j in 2:K
        y[starting_idx] = atanh(W[1, j])
        starting_idx += 1
        remainder_sq = W[j, j]^2
        for i in (j - 1):-1:2
            idx = starting_idx + i - 2
            z = W[i, j] / sqrt(remainder_sq)
            y[idx] = asinh(z)
            remainder_sq += W[i, j]^2
        end
        starting_idx += length((j - 1):-1:2)
    end

    return y
end

_link_chol_lkj_from_lower(W::AbstractMatrix) = _link_chol_lkj_from_upper(transpose_eager(W))

"""
    _inv_link_chol_lkj(y)

Inverse link function for cholesky factor.
"""
function _inv_link_chol_lkj(Y::AbstractMatrix)
    LinearAlgebra.require_one_based_indexing(Y)
    K = LinearAlgebra.checksquare(Y)

    W = similar(Y)
    T = float(eltype(W))
    logJ = zero(T)

    @inbounds for j in 1:K
        log_remainder = zero(T)  # log of proportion of unit vector remaining
        for i in 1:(j - 1)
            z = tanh(Y[i, j])
            W[i, j] = z * exp(log_remainder)
            log_remainder -= LogExpFunctions.logcosh(Y[i, j])
            logJ += log_remainder
        end
        logJ += log_remainder
        W[j, j] = exp(log_remainder)
        for i in (j + 1):K
            W[i, j] = 0
        end
    end

    return W, logJ
end

function _inv_link_chol_lkj(y::AbstractVector)
    LinearAlgebra.require_one_based_indexing(y)
    K = _triu1_dim_from_length(length(y))

    W = similar(y, K, K)
    T = float(eltype(W))
    logJ = zero(T)

    z_vec = map(tanh, y)
    lc_vec = map(LogExpFunctions.logcosh, y)

    idx = 1
    @inbounds for j in 1:K
        log_remainder = zero(T)  # log of proportion of unit vector remaining
        for i in 1:(j - 1)
            z = z_vec[idx]
            W[i, j] = z * exp(log_remainder)
            log_remainder -= lc_vec[idx]
            logJ += log_remainder
            idx += 1
        end
        logJ += log_remainder
        W[j, j] = exp(log_remainder)
        for i in (j + 1):K
            W[i, j] = 0
        end
    end

    return W, logJ
end

# shared reverse-mode AD rule code
function _inv_link_chol_lkj_rrule(y::AbstractVector)
    LinearAlgebra.require_one_based_indexing(y)
    K = _triu1_dim_from_length(length(y))

    W = similar(y, K, K)
    T = typeof(log(one(eltype(W))))
    logJ = zero(T)

    z_vec = map(tanh, y)
    lc_vec = map(LogExpFunctions.logcosh, y)

    idx = 1
    W[1, 1] = 1
    @inbounds for j in 2:K
        log_remainder = zero(T)  # log of proportion of unit vector remaining
        for i in 1:(j - 1)
            z = z_vec[idx]
            W[i, j] = z * exp(log_remainder)
            log_remainder -= lc_vec[idx]
            logJ += log_remainder
            idx += 1
        end
        logJ += log_remainder
        W[j, j] = exp(log_remainder)
        for i in (j + 1):K
            W[i, j] = 0
        end
    end

    function pullback_inv_link_chol_lkj((ΔW, ΔlogJ))
        LinearAlgebra.require_one_based_indexing(ΔW)
        Δy = similar(y)

        idx_local = lastindex(y)
        @inbounds for j in K:-1:2
            Δlog_remainder = W[j, j] * ΔW[j, j] + 2ΔlogJ
            for i in (j - 1):-1:1
                W_ΔW = W[i, j] * ΔW[i, j]
                z = z_vec[idx_local]
                Δy[idx_local] = (inv(z) - z) * W_ΔW - z * Δlog_remainder
                idx_local -= 1
                Δlog_remainder += ΔlogJ + W_ΔW
            end
        end

        return Δy
    end

    return (W, logJ), pullback_inv_link_chol_lkj
end

function _inv_link_chol_lkj_rrule(y::AbstractMatrix)
    K = LinearAlgebra.checksquare(y)
    y_vec = Bijectors._triu_to_vec(y, 1)
    W_logJ, back = _inv_link_chol_lkj_reverse(y_vec)
    function pullback_inv_link_chol_lkj(ΔW_ΔlogJ)
        return update_triu_from_vec(_triu_to_vec(back(ΔW_ΔlogJ), 1), 1, K)
    end

    return W_logJ, pullback_inv_link_chol_lkj
end

function _logabsdetjac_inv_corr(Y::AbstractMatrix)
    K = LinearAlgebra.checksquare(Y)

    result = float(zero(eltype(Y)))
    @inbounds for j in 2:K, i in 1:(j - 1)
        result -= (K - i + 1) * LogExpFunctions.logcosh(Y[i, j])
    end
    return result
end

function _logabsdetjac_inv_corr(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    for (i, y_i) in enumerate(y)
        row_idx = vec_to_triu1_row_index(i)
        result -= (K - row_idx + 1) * LogExpFunctions.logcosh(y_i)
    end
    return result
end

function _logabsdetjac_inv_chol(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    idx = 1
    @inbounds for j in 2:K
        tmp = zero(result)
        for _ in 1:(j - 1)
            logcoshy = LogExpFunctions.logcosh(y[idx])
            tmp -= logcoshy
            result += tmp - logcoshy
            idx += 1
        end
    end

    return result
end
