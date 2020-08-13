# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

struct CorrBijector <: Bijector{2} end

function (b::CorrBijector)(x::AbstractMatrix{<:Real})    
    w = cholesky(x).U + zero(x) # convert to dense matrix
    r = link_w_lkj(w) 
    return r
end

(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

function (ib::Inverse{<:CorrBijector})(y::AbstractMatrix{<:Real})
    w = inv_link_w_lkj(y)
    return w' * w
end
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


function logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})
    @assert size(y, 1) == size(y, 2)
    K = size(y, 1)
    
    left = zero(eltype(y)) # Initial summand may make looping looks weird :(
    @inbounds for j=2:K, i=1:(j-1)
        # left += (K-i-1) * log(1 - tanh(y[i, j])^2) # lacks numerically stable
        left += (K-i-1) * 2 * (log(2) - log(exp(y[i,j]) + exp(-y[i,j])))
    end
    
    right = zero(eltype(y))
    @inbounds for j=2:K, i=1:(j-1)
        right += log(cosh(y[i, j])^2)
    end
    
    return left / 2 - right
end
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    
    return -logabsdetjac(inv(b),(b(X))) # It may be more efficient if we can use un-contraint value to prevent call of b
end
logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end


function inv_link_w_lkj(y)
    @assert size(y, 1) == size(y, 2)
    K = size(y, 1)

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

function link_w_lkj(w)
    @assert size(w, 1) == size(w, 2)
    K = size(w, 1)

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
