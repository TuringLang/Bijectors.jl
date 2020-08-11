# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

struct CorrBijector <: Bijector{2} end

(b::CorrBijector)(X::AbstractMatrix{<:Real}) = link_lkj(X)
(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

(ib::Inverse{<:CorrBijector})(Y::AbstractMatrix{<:Real}) = inv_link_lkj(Y)
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real}) = log_abs_det_jac_lkj(y)
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    
    return -log_abs_det_jac_lkj(b(X)) # It may be more efficient if we can use un-contraint value to prevent call of b
end
logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end


function log_abs_det_jac_lkj(y)
    # it's defined on inverse mapping
    K = size(y, 1)
    
    z = tanh.(y)
    left = eltype(y)(0)
    for i = 1:(K-1), j = (i+1):K
        left += (K-i-1) * log(1 - z[i, j]^2)
    end
    
    right = eltype(y)(0)
    for i = 1:(K-1), j = (i+1):K
        right += log(cosh(y[i, j])^2)
    end
    
    return left / 2 - right
end

function inv_link_w_lkj(y)
    K = size(y, 1)

    z = tanh.(y)
    w = similar(z)
    
    w[1,1] = 1
    for j in 1:K
        w[1, j] = 1
    end

    for i in 2:K
        for j in 1:(i-1)
            w[i, j] = 0
        end
        for j in i:K
            w[i, j] = w[i-1, j] * sqrt(1 - z[i-1, j]^2)
        end
    end

    for i in 1:K
        for j in (i+1):K
            w[i, j] = w[i, j] * z[i, j]
        end
    end
    
    return w
end

function inv_link_lkj(y)
    w = inv_link_w_lkj(y)
    return w' * w
end

function link_w_lkj(w)
    K = size(w, 1)

    z = zero(w)
    
    for j=2:K
        z[1, j] = w[1, j]
    end

    #=
    # This implementation will not work when w[i-1, j] = 0.
    # Though it is a zero measure set, unit matrix initialization will not work.

    for i=2:K, j=(i+1):K
        z[i, j] = w[i, j] / w[i-1, j] * z[i-1, j] / sqrt(1 - z[i-1, j]^2)
    end
    =#
    for i=2:K, j=(i+1):K
        p = w[i, j]
        for ip in 1:(i-1)
            p /= sqrt(1-z[ip, j]^2)
        end
        z[i,j] = p
    end
    
    y = atanh.(z)
    return y
end

function link_lkj(x)
    w = cholesky(x).U + zero(x) # convert to dense matrix
    r = link_w_lkj(w) 
    return r
end
