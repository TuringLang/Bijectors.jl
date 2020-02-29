####################
# Simplex bijector #
####################
struct SimplexBijector{T} <: Bijector{1} where {T} end
SimplexBijector() = SimplexBijector{true}()

function _clamp(x::T, b::Union{SimplexBijector, Inverse{<:SimplexBijector}}) where {T}
    bounds = (zero(T), one(T))
    clamped_x = clamp(x, bounds...)
    DEBUG && _debug("x = $x, bounds = $bounds, clamped_x = $clamped_x")
    return clamped_x
end

(b::SimplexBijector)(x::AbstractVector) = _simplex_bijector(x, b)
(b::SimplexBijector)(y::AbstractVector, x::AbstractVector) = _simplex_bijector!(y, x, b)
function _simplex_bijector(x::AbstractVector, b::SimplexBijector)
    return _simplex_bijector!(similar(x), x, b)
end
function _simplex_bijector!(y, x::AbstractVector, ::SimplexBijector{proj}) where {proj}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(x)
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
function (b::SimplexBijector)(X::AbstractMatrix)
    _simplex_bijector(X, b)
end
function (b::SimplexBijector)(
    Y::AbstractMatrix,
    X::AbstractMatrix,
)
    _simplex_bijector!(Y, X, b)
end
function _simplex_bijector(X::AbstractMatrix, b::SimplexBijector)
    _simplex_bijector!(similar(X), X, b)
end
function _simplex_bijector!(Y, X::AbstractMatrix, ::SimplexBijector{proj}) where {proj}
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(X)
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

function (ib::Inverse{<:SimplexBijector{proj}})(y::AbstractVector{T}) where {T, proj}
    _simplex_inv_bijector(y, ib.orig)
end
function (ib::Inverse{<:SimplexBijector})(
    x::AbstractVector{T},
    y::AbstractVector{T},
) where {T}
    _simplex_inv_bijector!(x, y, ib.orig)
end
function _simplex_inv_bijector(y::AbstractVector, b::SimplexBijector)
    return _simplex_inv_bijector!(similar(y), y, b)
end
function _simplex_inv_bijector!(x, y::AbstractVector, b::SimplexBijector{proj}) where {proj}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(y)
    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), b)
    sum_tmp = zero(T)
    @inbounds @simd for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, b)
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        x[K] = _clamp(one(T) - sum_tmp, b)
    else
        x[K] = _clamp(one(T) - sum_tmp - y[K], b)
    end
    
    return x
end

# Vectorised implementation of the above.
function (ib::Inverse{<:SimplexBijector})(Y::AbstractMatrix)
    _simplex_inv_bijector(Y, ib.orig)
end
function (ib::Inverse{<:SimplexBijector})(
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
) where {T <: Real}
    _simplex_inv_bijector!(X, Y, ib.orig)
end
function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector)
    _simplex_inv_bijector!(similar(Y), Y, b)
end
function _simplex_inv_bijector!(X, Y::AbstractMatrix, b::SimplexBijector{proj}) where {proj}
    K, N = size(Y, 1), size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(Y)
    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp, z = zero(T), StatsFuns.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), b)
        for k in 2:(K - 1)
            z = StatsFuns.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, b)
        end
        sum_tmp += X[K - 1, n]
        if proj
            X[K, n] = _clamp(one(T) - sum_tmp, b)
        else
            X[K, n] = _clamp(one(T) - sum_tmp - Y[K, n], b)
        end
    end

    return X
end

function logabsdetjac(b::SimplexBijector, x::AbstractVector{T}) where {T}
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
#=
function logabsdetjac_val_gradient(b::SimplexBijector, x::AbstractVector)
    T = eltype(x)
    ϵ = _eps(T)
    lp = zero(T)
    
    K = length(x)

    g = similar(x)
    g .= 0
    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    lp += log(z + ϵ) + log((one(T) + ϵ) - z)
    g[1] = -1/(z + ϵ) + 1/((one(T) + ϵ) - z)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k-1]
        z = x[k] / ((one(T) + ϵ) - sum_tmp)
        lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
        g[k-1] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * x[k] / ((one(T) + ϵ) - sum_tmp)^2 - 1
        g[k] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * 1 / ((one(T) + ϵ) - sum_tmp)
    end
    return -lp, g
end
=#
function simplex_logabsdetjac_gradient(x::AbstractVector)
    T = eltype(x)
    ϵ = _eps(T)    
    K = length(x)
    g = similar(x)
    g .= 0
    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    g[1] = -1/(z + ϵ) + 1/((one(T) + ϵ) - z)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k-1]
        z = x[k] / ((one(T) + ϵ) - sum_tmp)
        g[k-1] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * x[k] / ((one(T) + ϵ) - sum_tmp)^2 - 1
        g[k] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * 1 / ((one(T) + ϵ) - sum_tmp)
    end
    return g
end

function logabsdetjac(b::SimplexBijector, x::AbstractMatrix{<:Real})
    mapvcat(eachcol(x)) do c
        logabsdetjac(b, c)
    end
end

#=
function simplex_link_val_adjoint(
    x::AbstractVector{T},
    Δ::AbstractVector,
    proj::Bool = true,
) where {T <: Real}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    y = similar(x)
    ϵ = _eps(T)
    sum_tmp = zero(T)
    nΔ = zeros(T, length(Δ))
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = StatsFuns.logit(z) + log(T(K - 1))
    @inbounds nΔ[1] = Δ[1] * (1/z + 1/(1-z)) * (one(T) - 2ϵ)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        y[k] = StatsFuns.logit(z) + log(T(K - k))
        nΔ[k] += Δ[k] * (1/z + 1/(1-z)) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        for i in 1:k-1
            nΔ[i] += Δ[k] * (1/z + 1/(1-z)) * (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)^2
        end
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        y[K] = zero(T)
    else
        y[K] = one(T) - sum_tmp - x[K]
        @simd for i in 1:K
            nΔ[i] += -Δ[K]
        end
    end

    return y, nΔ
end

function simplex_link_val_jacobian(
    x::AbstractVector{T},
    proj::Bool = true,
) where {T <: Real}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    y = similar(x)
    dydxt = similar(x, length(x), length(x))
    @inbounds dydxt .= 0
    ϵ = _eps(T)
    sum_tmp = zero(T)

    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = StatsFuns.logit(z) + log(T(K - 1))
    @inbounds dydxt[1,1] = (1/z + 1/(1-z)) * (one(T) - 2ϵ)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        y[k] = StatsFuns.logit(z) + log(T(K - k))
        dydxt[k,k] = (1/z + 1/(1-z)) * (one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        for i in 1:k-1
            dydxt[i,k] = (1/z + 1/(1-z)) * (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)^2
        end
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        y[K] = zero(T)
    else
        y[K] = one(T) - sum_tmp - x[K]
        @simd for i in 1:K
            dydxt[i,K] = -1
        end
    end

    return y, UpperTriangular(dydxt)'
end
=#
function simplex_link_jacobian(
    x::AbstractVector{T},
    proj::Bool = true,
) where {T <: Real}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
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

#=
function simplex_invlink_val_adjoint(
    y::AbstractVector{T},
    Δ::AbstractVector,
    proj::Bool = true,
) where {T <: Real}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"
    x = similar(y)
    nΔ = similar(Δ)
    @inbounds nΔ .= 0
    dxdy = similar(y, length(y), length(y))
    @inbounds dxdy .= 0

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, SimplexBijector())
    @inbounds x[1] = clamped_x
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
        nΔ[1] = Δ[1] * dxdy[1,1]
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, SimplexBijector())
        x[k] = clamped_x
        if unclamped_x == clamped_x
            dxdy[k,k] = z * (1 - z) * ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ)
            nΔ[k] = Δ[k] * dxdy[k,k]
            for i in 1:k-1
                for j in i:k-1
                    temp = -dxdy[j,i] * z / (one(T) - 2ϵ)
                    dxdy[k,i] += temp
                    nΔ[i] += Δ[k] * temp
                end
            end
        end
    end
    @inbounds sum_tmp += clamped_x
    @inbounds if proj
    	unclamped_x = one(T) - sum_tmp
        clamped_x = _clamp(unclamped_x, SimplexBijector())
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, SimplexBijector())
        if unclamped_x == clamped_x
            nΔ[K] = -Δ[K]
        end
    end
    x[K] = clamped_x
    @inbounds if unclamped_x == clamped_x
        for i in 1:K-1
            @simd for j in i:K-1
                nΔ[i] += -Δ[K] * dxdy[j,i]
            end
        end
    end
    return x, nΔ
end

function simplex_invlink_val_jacobian(
    y::AbstractVector{T},
    proj::Bool = true,
) where {T <: Real}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"
    x = similar(y)
    dxdy = similar(y, length(y), length(y))
    @inbounds dxdy .= 0

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, SimplexBijector())
    @inbounds x[1] = clamped_x
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, SimplexBijector())
        x[k] = clamped_x
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
        clamped_x = _clamp(unclamped_x, SimplexBijector())
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, SimplexBijector())
        if unclamped_x == clamped_x
            dxdy[K,K] = -1
        end
    end
    x[K] = clamped_x
    @inbounds if unclamped_x == clamped_x
        for i in 1:K-1
            @simd for j in i:K-1
                dxdy[K,i] += -dxdy[j,i]
            end
        end
    end
    return x, LowerTriangular(dxdy)
end
=#
function simplex_invlink_jacobian(
    y::AbstractVector{T},
    proj::Bool = true,
) where {T <: Real}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"
    dxdy = similar(y, length(y), length(y))
    @inbounds dxdy .= 0

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, SimplexBijector())
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, SimplexBijector())
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
        clamped_x = _clamp(unclamped_x, SimplexBijector())
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, SimplexBijector())
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

function jacobian(ib::Inverse{<:SimplexBijector{proj}}, y::AbstractVector{T}) where {proj, T}
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
