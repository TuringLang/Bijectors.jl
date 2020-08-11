# See stan doc for parametrization method:
# https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html
# (7/30/2020) their "manageable expression" is wrong...

function upper1(AT, A)
    # @show typeof(AT) typeof(A) AT A # TODO: remove this line before committing
    AU = zero(AT)
    for i=1:size(A,1), j=(i+1):size(A,2)
        AU[i,j] = A[i,j]
    end
    return AU
end

struct CorrBijector <: Bijector{2} end

(b::CorrBijector)(X::AbstractMatrix{<:Real}) = link_lkj(X)
(b::CorrBijector)(X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, X)

(ib::Inverse{<:CorrBijector})(Y::AbstractMatrix{<:Real}) = inv_link_lkj(Y)
(ib::Inverse{<:CorrBijector})(Y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, Y)


logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real}) = log_abs_det_jac_lkj(y)
function logabsdetjac(b::CorrBijector, X::AbstractMatrix{<:Real})
    #=
    if !LinearAlgebra.isposdef(X)
        println("!isposdef(X)")
        return NaN # prevent Cholesky decomposition to break inference
    end
    =#
    
    return -log_abs_det_jac_lkj(b(X)) # It may be more efficient if we can use un-contraint value to prevent call of b
end
logabsdetjac(b::CorrBijector, X::AbstractArray{<:AbstractMatrix{<:Real}}) = mapvcat(X) do x
    logabsdetjac(b, x)
end


function log_abs_det_jac_lkj(y)
    # it's defined on inverse mapping
    K = size(y, 1)
    
    z = tanh.(y)
    left = 0
    for i = 1:(K-1), j = (i+1):K
        left += (K-i-1) * log(1 - z[i, j]^2)
    end
    
    right = 0
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

    # `zero` isn't compatible with ReverseDiff, so we use `similar`
    #=
    z = similar(w)
    for i=1:K, j=1:K
        z[i,j] = 0
    end
    =#
    z = zero(w)
    # @show typeof(w) w
    # @show typeof(z) z # TODO: remove this line before committing
    
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
            p *= 1 / sqrt(1-z[ip, j]^2)
        end
        z[i,j] = p
    end
    
    y = atanh.(z)
    return y
end

function link_lkj(x)
    w = cholesky(x).U + zero(x) # promote
    # w = collect(cholesky(x).U)
    # w = convert(typeof(x), cholesky(x).U) # ? test requires it, such quirk
    # w = upper(parent(cholesky(x).U))
    # return link_w_lkj(w)
    # @show typeof(x) x # TODO: remove them before release
    # @show typeof(w) w
    r = link_w_lkj(w) 
    # return r - lower(parent(r)) # test requires it, such quirk
    # @show typeof(x) typeof(r)
    # return upper1(x, r)
    return r # promote?
end
