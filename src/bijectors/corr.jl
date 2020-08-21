# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

struct CorrBijector <: Bijector{2} end

function (b::CorrBijector)(x::AbstractMatrix{<:Real})    
    w = cholesky(x).U + zero(x) # convert to dense matrix
    r = _link_chol_lkj(w) 
    return r
end

(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

function (ib::Inverse{<:CorrBijector})(y::AbstractMatrix{<:Real})
    w = _inv_link_chol_lkj(y)
    return w' * w
end
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


function logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})
    K = LinearAlgebra.checksquare(y)
    
    result = float(zero(eltype(y))
    for j in 2:K, i in 1:(j - 1)
        @inbounds abs_y_i_j = abs(y[i, j])
        result += (K - i + 1) * (logtwo - (abs_y_i_j + log1pexp(-2 * abs_y_i_j)))
    end
    
    return result
end
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    return -logabsdetjac(inv(b),(b(X))) # It may be more efficient if we can use un-contraint value to prevent call of b
end
function logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}})
    return mapvcat(X) do x
        logabsdetjac(b, x)
    end
end


function _inv_link_chol_lkj(y)
    K = LinearAlgebra.checksquare(y)

    z = tanh.(y)
    w = similar(z)
    
    w[1,1] = 1
    @inbounds for j in 1:K
        w[1, j] = 1
    end

    @inbounds for j in 1:K
        for i in j+1:K
            w[i, j] = 0
        end
        for i in 2:j
            w[i, j] = w[i-1, j] * sqrt(1 - z[i-1, j]^2)
        end
    end

    @inbounds for j in 2:K
        for i in 1:j-1
            w[i, j] = w[i, j] * z[i, j]
        end
    end
    
    return w
end

function _link_chol_lkj(w)
    K = LinearAlgebra.checksquare(w)

    z = zero(w)
    
    @inbounds for j=2:K
        z[1, j] = w[1, j]
    end

    #=
    # This implementation will not work when w[i-1, j] = 0.
    # Though it is a zero measure set, unit matrix initialization will not work.

    for i=2:K, j=(i+1):K
        z[i, j] = (w[i, j] / w[i-1, j]) * (z[i-1, j] / sqrt(1 - z[i-1, j]^2))
    end
    For `(i, j)` in the loop below, we define

        z₍ᵢ₋₁, ⱼ₎ = w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² (1 / √(1 - z₍ₖ,ⱼ₎²))

    and so

        z₍ᵢ,ⱼ₎ = w₍ᵢ,ⱼ₎ * ∏ₖ₌₁ⁱ⁻¹ (1 / √(1 - z₍ₖ,ⱼ₎²))
               = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²))
               = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²)) / w₍ᵢ₋₁,ⱼ₎
               = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (z₍ᵢ₋₁,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎)
               = (w₍ᵢ,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎) * (z₍ᵢ₋₁,ⱼ₎ / √(1 - z₍ᵢ₋₁,ⱼ₎²))

    which is the above implementation.
    =#
    @inbounds for j=3:K, i=2:j-1
        p = w[i, j]
        for ip in 1:(i-1)
            p /= sqrt(1-z[ip, j]^2)
        end
        z[i,j] = p
    end
    
    y = atanh.(z)
    return y
end
