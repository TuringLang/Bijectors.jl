"""
    CorrBijector <: Bijector{2}

A bijector implementation of Stan's parametrization method for Correlation matrix:
https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html

Note:(7/30/2020) their "manageable expression" is wrong, used expression is derived from 
scratch.
"""
struct CorrBijector <: Bijector{2} end

function (b::CorrBijector)(x::AbstractMatrix{<:Real})    
    w = cholesky(x).U  # keep LowerTriangular until here can avoid some computation
    r = _link_chol_lkj(w) 
    return r + zero(x) 
    # This dense format itself is required by a test, though I can't get the point.
    # https://github.com/TuringLang/Bijectors.jl/blob/b0aaa98f90958a167a0b86c8e8eca9b95502c42d/test/transform.jl#L67
end

(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

function (ib::Inverse{<:CorrBijector})(y::AbstractMatrix{<:Real})
    w = _inv_link_chol_lkj(y)
    return w' * w
end
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


function logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})
    K = LinearAlgebra.checksquare(y)
    
    result = float(zero(eltype(y)))
    for j in 2:K, i in 1:(j - 1)
        @inbounds abs_y_i_j = abs(y[i, j])
        result += (K - i + 1) * (logtwo - (abs_y_i_j + log1pexp(-2 * abs_y_i_j)))
    end
    
    return result
end
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    #=
    It may be more efficient if we can use un-contraint value to prevent call of b
    It's recommended to directly call 
    `logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})`
    if possible.
    =#
    return -logabsdetjac(inv(b), (b(X))) 
end
function logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}})
    return mapvcat(X) do x
        logabsdetjac(b, x)
    end
end


function _inv_link_chol_lkj(y)
    K = LinearAlgebra.checksquare(y)

    w = similar(y)
    
    @inbounds for j in 1:K
        w[1, j] = 1
        for i in 2:j
            z = tanh(y[i-1, j])
            tmp = w[i-1, j]
            w[i-1, j] = z * tmp
            w[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j+1):K
            w[i, j] = 0
        end
    end
    
    return w
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

For equivelence, following explanations is given by @torfjelde:

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
function _link_chol_lkj(w)
    K = LinearAlgebra.checksquare(w)

    z = similar(w) # z is also UpperTriangular. 
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    # This block can't be integrated with loop below, because w[1,1] != 0.
    @inbounds z[1, 1] = 0

    @inbounds for j=2:K
        z[1, j] = w[1, j]
        tmp = sqrt(1 - z[1, j]^2)
        for i in 2:(j - 1)
            p = w[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            z[i, j] = p
        end
        z[j, j] = 0
    end
    
    z .= atanh.(z)
    return z
end
