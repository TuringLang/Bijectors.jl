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
    w = cholesky_upper(X)
    r = _link_chol_lkj(w)
    return r
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

function transform(::Inverse{VecCorrBijector}, y::AbstractVector{<:Real})
    return pd_from_upper(_inv_link_chol_lkj(y))
end

function logabsdetjac(::Inverse{VecCorrBijector}, y::AbstractVector{<:Real})
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

function transform(b::Inverse{VecCholeskyBijector}, y::AbstractVector{<:Real})
    if b.orig.mode === :U
        # This Cholesky constructor is compatible with Julia v1.6
        # for later versions Cholesky(::UpperTriangular) works
        return Cholesky(_inv_link_chol_lkj(y), 'U', 0)
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        # HACK: Need to make materialize the transposed matrix to avoid numerical instabilities.
        # If we don't, the return-type can be both `Matrix` and `Transposed`.
        return Cholesky(transpose_eager(_inv_link_chol_lkj(y)), 'L', 0)
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

    z = similar(W) # z is also UpperTriangular. 
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    # This block can't be integrated with loop below, because W[1,1] != 0.
    @inbounds z[:, 1] .= 0

    @inbounds for j in 2:K
        z[1, j] = atanh(W[1, j])
        tmp = sqrt(1 - W[1, j]^2)
        for i in 2:(j - 1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            z[i, j] = atanh(p)
        end
        for i in j:K
            z[i, j] = 0
        end
    end

    return z
end

function _link_chol_lkj_from_upper(W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)
    N = ((K - 1) * K) ÷ 2   # {K \choose 2} free parameters

    z = similar(W, N)

    idx = 1
    @inbounds for j in 2:K
        z[idx] = atanh(W[1, j])
        idx += 1
        tmp = sqrt(1 - W[1, j]^2)
        for i in 2:(j - 1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            z[idx] = atanh(p)
            idx += 1
        end
    end

    return z
end

_link_chol_lkj_from_lower(W::AbstractMatrix) = _link_chol_lkj_from_upper(transpose_eager(W))

"""
    _inv_link_chol_lkj(y)

Inverse link function for cholesky factor.
"""
function _inv_link_chol_lkj(Y::AbstractMatrix)
    K = LinearAlgebra.checksquare(Y)

    W = similar(Y)

    @inbounds for j in 1:K
        W[1, j] = 1
        for i in 2:j
            z = tanh(Y[i - 1, j])
            tmp = W[i - 1, j]
            W[i - 1, j] = z * tmp
            W[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j + 1):K
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
            tmp = W[i - 1, j]
            W[i - 1, j] = z * tmp
            W[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j + 1):K
            W[i, j] = 0
        end
    end

    return W
end

function _logabsdetjac_inv_corr(Y::AbstractMatrix)
    K = LinearAlgebra.checksquare(Y)

    result = float(zero(eltype(Y)))
    for j in 2:K, i in 1:(j - 1)
        @inbounds abs_y_i_j = abs(Y[i, j])
        result +=
            (K - i + 1) * (
                IrrationalConstants.logtwo -
                (abs_y_i_j + LogExpFunctions.log1pexp(-2 * abs_y_i_j))
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
        result +=
            (K - row_idx + 1) * (
                IrrationalConstants.logtwo -
                (abs_y_i + LogExpFunctions.log1pexp(-2 * abs_y_i))
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
        for _ in 1:(j - 1)
            z = tanh(y[idx])
            logz = log(1 - z^2)
            result += logz + (tmp / 2)
            tmp += logz
            idx += 1
        end
    end

    return result
end
