# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

struct CorrBijector <: Bijector{2} end

(b::CorrBijector)(X::AbstractMatrix{<:Real}) = link_lkj(X)
(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

(ib::Inverse{<:CorrBijector})(Y::AbstractMatrix{<:Real}) = inv_link_lkj(Y)
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)

logabsdetjac(::CorrBijector, X::AbstractMatrix{<:Real}) = log_abs_det_jac_lkj(X)
# logabsdetjac(b::CorrBijector, Xcf::Cholesky)
logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end

function log_abs_det_jac_lkj(y)
    K = size(y, 1)
    
    z = tanh.(y)
    left = 0
    for i = 1:(K-1), j = (i+1):K
        left += (K-i-1) * log(1 - z[i, j]^2)
    end
    
    right = 0
    for i = 1:(K-1), j = (i+1):K
        right += 1 / cosh(y[i, j])^2
    end
    
    return 0.5 * left + right
end

function inv_link_w_lkj(y)
    z = tanh.(y)
    w = similar(z)
    
    w[1,1] = 1
    for j in 1:size(w, 2)
        w[1, j] = 1
    end

    for i in 2:size(w, 1)
        for j in 1:(i-1)
            w[i, j] = 0
        end
        for j in i:size(w, 2)
            w[i, j] = w[i-1, j] * sqrt(1 - z[i-1, j]^2)
        end
    end

    for i in 1:size(w, 1)
        for j in (i+1):size(w, 2)
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
    z = zeros(size(w)...)
    
    for j=2:size(z, 2)
        z[1, j] = w[1, j]
    end

    for i=2:size(z, 2), j=(i+1):size(z, 2)
        z[i, j] = w[i, j] / w[i-1, j] * z[i-1, j] / sqrt(1 - z[i-1, j]^2)
    end
    
    y = atanh.(z)
    return y
end

function link_lkj(x)
    w = cholesky(x).U
    return link_w_lkj(w)
end
