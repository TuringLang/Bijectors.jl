####################
# Simplex bijector #
####################
struct SimplexBijector{T} <: Bijector{1} where {T} end
SimplexBijector() = SimplexBijector{true}()

function _clamp(x::T, b::Union{SimplexBijector, Inversed{<:SimplexBijector}}) where {T}
    bounds = (zero(T), one(T))
    clamped_x = clamp(x, bounds...)
    DEBUG && @debug "x = $x, bounds = $bounds, clamped_x = $clamped_x"
    return clamped_x
end

(b::SimplexBijector)(x::AbstractVecOrMat) = _simplex_bijector(b, x)

function _simplex_bijector(b::SimplexBijector{proj}, x::AbstractVector{T}) where {T, proj}
    y, K = similar(x), length(x)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    sum_tmp = zero(T)
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = StatsFuns.logit(z) + log(T(K - 1))
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        y[k] = StatsFuns.logit(z) + log(T(K - k))
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        y[K] = zero(T)
    else
        y[K] = one(T) - sum_tmp - x[K]
    end

    return y
end

# Vectorised implementation of the above.
function _simplex_bijector(b::SimplexBijector{proj}, X::AbstractMatrix{T}) where {T, proj}
    Y, K, N = similar(X), size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp = zero(T)
        z = X[1, n] * (one(T) - 2ϵ) + ϵ
        Y[1, n] = StatsFuns.logit(z) + log(T(K - 1))
        for k in 2:(K - 1)
            sum_tmp += X[k - 1, n]
            z = (X[k, n] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
            Y[k, n] = StatsFuns.logit(z) + log(T(K - k))
        end
        sum_tmp += X[K-1, n]
        if proj
            Y[K, n] = zero(T)
        else
            Y[K, n] = one(T) - sum_tmp - X[K, n]
        end
    end

    return Y
end

# Inverse
(ib::Inversed{<:SimplexBijector})(y::AbstractVecOrMat) = _simplex_bijector_inv(ib, y)

function _simplex_bijector_inv(ib::Inversed{<:SimplexBijector{proj}}, y::AbstractVector{T}) where {T, proj}
    x, K = similar(y), length(y)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), ib.orig)
    sum_tmp = zero(T)
    @inbounds @simd for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, ib.orig)
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        x[K] = _clamp(one(T) - sum_tmp, ib.orig)
    else
        x[K] = _clamp(one(T) - sum_tmp - y[K], ib.orig)
    end
    
    return x
end

# Vectorised implementation of the above.
function _simplex_bijector_inv(
    ib::Inversed{<:SimplexBijector{proj}}, Y::AbstractMatrix{T}
) where {T<:Real, proj}
    X, K, N = similar(Y), size(Y, 1), size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp, z = zero(T), StatsFuns.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), ib.orig)
        for k in 2:(K - 1)
            z = StatsFuns.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, ib.orig)
        end
        sum_tmp += X[K - 1, n]
        if proj
            X[K, n] = _clamp(one(T) - sum_tmp, ib.orig)
        else
            X[K, n] = _clamp(one(T) - sum_tmp - Y[K, n], ib.orig)
        end
    end

    return X
end


function logabsdetjac(b::SimplexBijector, x::AbstractVector{T}) where T
    ϵ = _eps(T)
    lp = zero(T)
    
    K = length(x)

    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    lp += log(z + ϵ) + log((one(T) + ϵ) - z)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k-1]
        z = x[k] / ((one(T) + ϵ) - sum_tmp)
        lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
    end

    return - lp
end

function logabsdetjac(b::SimplexBijector, x::AbstractMatrix{<:Real})
    return [logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)]
end

# jacobian
function jacobian(b::SimplexBijector{proj}, x::AbstractVector{T}) where {proj, T}
    K = length(x)
    dydxt = similar(x, length(x), length(x))
    @inbounds dydxt .= 0
    ϵ = _eps(T)
    sum_tmp = zero(T)

    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds dydxt[1,1] = (1/z + 1/(1-z)) * (one(T) - 2ϵ)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        dydxt[k,k] = (1/z + 1/(1-z)) * (one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        for i in 1:k-1
            dydxt[i,k] = (1/z + 1/(1-z)) * (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)^2
        end
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if !proj
        @simd for i in 1:K
            dydxt[i,K] = -1
        end
    end

    return UpperTriangular(dydxt)'
end

function jacobian(ib::Inversed{<:SimplexBijector{proj}}, y::AbstractVector{T}) where {proj, T}
    b = ib.orig
    
    K = length(y)
    dxdy = similar(y, length(y), length(y))
    @inbounds dxdy .= 0

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, b)
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, b)
        if unclamped_x == clamped_x
            dxdy[k,k] = z * (1 - z) * ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ)
            for i in 1:k-1
                for j in i:k-1
                    dxdy[k,i] += -dxdy[j,i] * z / (one(T) - 2ϵ)
                end
            end
        end
    end
    @inbounds sum_tmp += clamped_x
    @inbounds if proj
    	unclamped_x = one(T) - sum_tmp
        clamped_x = _clamp(unclamped_x, b)
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, b)
        if unclamped_x == clamped_x
            dxdy[K,K] = -1
        end
    end
    @inbounds if unclamped_x == clamped_x
        for i in 1:K-1
            @simd for j in i:K-1
                dxdy[K,i] += -dxdy[j,i]
            end
        end
    end
    return LowerTriangular(dxdy)
end
