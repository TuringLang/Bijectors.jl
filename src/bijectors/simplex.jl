####################
# Simplex bijector #
####################
struct SimplexBijector{T} <: Bijector{1} where {T} end
SimplexBijector() = SimplexBijector{true}()

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
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
    sum_tmp = zero(T)
    @inbounds @simd for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        x[K] = _clamp(one(T) - sum_tmp, 0, 1)
    else
        x[K] = _clamp(one(T) - sum_tmp - y[K], 0, 1)
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
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
        for k in 2:(K - 1)
            z = StatsFuns.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
        end
        sum_tmp += X[K - 1, n]
        if proj
            X[K, n] = _clamp(one(T) - sum_tmp, 0, 1)
        else
            X[K, n] = _clamp(one(T) - sum_tmp - Y[K, n], 0, 1)
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

    return -lp
end
function simplex_logabsdetjac_gradient(x::AbstractVector)
    T = eltype(x)
    ϵ = _eps(T)    
    K = length(x)
    g = similar(x)
    g .= 0
    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    #lp += log(z + ϵ) + log((one(T) + ϵ) - z)
    g[1] = -1/(z + ϵ) + 1/((one(T) + ϵ) - z)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k-1]
        temp = ((one(T) + ϵ) - sum_tmp)
        z = x[k] / temp
        #lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log(temp)
        dzdx = 1 / temp
        dldz = (1/(z + ϵ) - 1/((one(T) + ϵ) - z))
        dldx = dldz * dzdx
	    g[k] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * 1 / ((one(T) + ϵ) - sum_tmp)
        for i in 1:k-1
	        dzdxp = x[k] * dzdx^2
	        dldxp = dldz * dzdxp - 1 / temp
	        g[i] -= dldxp
	    end
    end
    return g
end
function logabsdetjac(b::SimplexBijector, x::AbstractMatrix{T}) where {T}
    ϵ = _eps(T)
    nlp = similar(x, T, size(x, 2))
    nlp .= zero(T)

    K = size(x, 1)
    for col in 1:size(x, 2)
        sum_tmp = zero(eltype(x))
        @inbounds z = x[1,col]
        nlp[col] -= log(z + ϵ) + log((one(T) + ϵ) - z)
        @inbounds @simd for k in 2:(K - 1)
            sum_tmp += x[k-1,col]
            z = x[k,col] / ((one(T) + ϵ) - sum_tmp)
            nlp[col] -= log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
        end
    end
    return nlp
end
function simplex_logabsdetjac_gradient(x::AbstractMatrix)
    T = eltype(x)
    ϵ = _eps(T)
    K = size(x, 1)
    g = similar(x)
    g .= 0
    @inbounds @simd for col in 1:size(x, 2)
        sum_tmp = zero(eltype(x))
        z = x[1,col]
        #lp += log(z + ϵ) + log((one(T) + ϵ) - z)
        g[1,col] = -1/(z + ϵ) + 1/((one(T) + ϵ) - z)
        for k in 2:(K - 1)
            sum_tmp += x[k-1,col]
            temp = ((one(T) + ϵ) - sum_tmp)
            z = x[k,col] / temp
            #lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log(temp)
            dzdx = 1 / temp
            dldz = (1/(z + ϵ) - 1/((one(T) + ϵ) - z))
            dldx = dldz * dzdx
            g[k,col] -= (1/(z + ϵ) - 1/((one(T) + ϵ) - z)) * 1 / ((one(T) + ϵ) - sum_tmp)
            for i in 1:k-1
                dzdxp = x[k,col] * dzdx^2
                dldxp = dldz * dzdxp - 1 / temp
                g[i,col] -= dldxp
            end
        end
    end
    return g
end

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
function jacobian(b::SimplexBijector{proj}, x::AbstractVector{T}) where {proj, T}
    return simplex_link_jacobian(x, proj)
end

#=
# This approach is faster for small random variables
# For larger ones, building the Jacobian and multiplying is faster because of BLAS.

function add_simplex_link_adjoint!(
    inΔ::AbstractVector,
    x::AbstractVector{T},
    outΔ::AbstractVector,
    proj::Bool = true,
) where {T <: Real}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    ϵ = _eps(T)
    sum_tmp = zero(T)
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds inΔ[1] += outΔ[1] * (1/z + 1/(1-z)) * (one(T) - 2ϵ)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
        inΔ[k] += outΔ[k] * (1/z + 1/(1-z)) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        for i in 1:k-1
            inΔ[i] += outΔ[k] * (1/z + 1/(1-z)) * (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)^2
        end
    end
    if !proj
        for i in 1:K
            inΔ[i] += -outΔ[K]
        end
    end
    return inΔ
end
function add_simplex_link_adjoint!(
    inΔ::AbstractMatrix,
    x::AbstractMatrix{T},
    outΔ::AbstractMatrix,
    proj::Bool = true,
) where {T <: Real}
    K = size(x, 1)
    @assert K > 1 "x needs to be of length greater than 1"
    ϵ = _eps(T)
    @inbounds for col in 1:size(x, 2)
        sum_tmp = zero(T)
        z = x[1,col] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
        inΔ[1,col] += outΔ[1,col] * (1/z + 1/(1-z)) * (one(T) - 2ϵ)
        @simd for k in 2:(K - 1)
            sum_tmp += x[k-1, col]
            # z ∈ [ϵ, 1-ϵ]
            # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
            z = (x[k,col] + ϵ)*(one(T) - 2ϵ)/((one(T) + ϵ) - sum_tmp)
            inΔ[k,col] += outΔ[k,col] * (1/z + 1/(1-z)) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
            for i in 1:k-1
                inΔ[i,col] += outΔ[k,col] * (1/z + 1/(1-z)) * (x[k,col] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)^2
            end
        end
        if !proj
            @simd for i in 1:K
                inΔ[i,col] += -outΔ[K,col]
            end
        end
    end
    return inΔ
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
    clamped_x = _clamp(unclamped_x, 0, 1)
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, 0, 1)
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
        clamped_x = _clamp(unclamped_x, 0, 1)
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, 0, 1)
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
function jacobian(ib::Inverse{<:SimplexBijector{proj}}, y::AbstractVector{T}) where {proj, T}
    return simplex_invlink_jacobian(y, proj)
end

#=
# This approach is faster for small random variables
# For larger ones, building the Jacobian and multiplying is faster because of BLAS.

function add_simplex_invlink_adjoint!(
    inΔ::AbstractVector,
    y::AbstractVector{T},
    outΔ::AbstractVector,
    proj::Bool = true,
) where {T <: Real}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"
    dxdy = similar(y, length(y), length(y))

    @inbounds dxdy .= 0
    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, 0, 1)
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
        inΔ[1] += outΔ[1] * dxdy[1,1]
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, 0, 1)
        if unclamped_x == clamped_x
            dxdy[k,k] = z * (1 - z) * ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ)
            inΔ[k] += outΔ[k] * dxdy[k,k]
            for i in 1:k-1
                for j in i:k-1
                    temp = -dxdy[j,i] * z / (one(T) - 2ϵ)
                    dxdy[k,i] += temp
                    inΔ[i] += outΔ[k] * temp
                end
            end
        end
    end
    @inbounds sum_tmp += clamped_x
    @inbounds if proj
    	unclamped_x = one(T) - sum_tmp
        clamped_x = _clamp(unclamped_x, 0, 1)
    else
    	unclamped_x = one(T) - sum_tmp - y[K]
        clamped_x = _clamp(unclamped_x, 0, 1)
        if unclamped_x == clamped_x
            inΔ[K] += -outΔ[K]
        end
    end
    @inbounds if unclamped_x == clamped_x
        for i in 1:K-1
            @simd for j in i:K-1
                inΔ[i] += -outΔ[K] * dxdy[j,i]
            end
        end
    end
    return inΔ
end
function add_simplex_invlink_adjoint!(
    inΔ::AbstractMatrix,
    y::AbstractMatrix{T},
    outΔ::AbstractMatrix,
    proj::Bool = true,
) where {T <: Real}
    K = size(y,1)
    @assert K > 1 "x needs to be of length greater than 1"
    dxdy = similar(y, size(y,1), size(y,1))

    @inbounds for col in 1:size(y,2)
        dxdy .= 0
        ϵ = _eps(T)
        z = StatsFuns.logistic(y[1,col] - log(T(K - 1)))
        unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
        clamped_x = _clamp(unclamped_x, 0, 1)
        if unclamped_x == clamped_x
            dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
            inΔ[1,col] += outΔ[1,col] * dxdy[1,1]
        end
        sum_tmp = zero(T)
        for k = 2:(K - 1)
            z = StatsFuns.logistic(y[k,col] - log(T(K - k)))
            sum_tmp += clamped_x
            unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
            clamped_x = _clamp(unclamped_x, 0, 1)
            if unclamped_x == clamped_x
                dxdy[k,k] = z * (1 - z) * ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ)
                inΔ[k,col] += outΔ[k,col] * dxdy[k,k]
                for i in 1:k-1
                    @simd for j in i:k-1
                        temp = -dxdy[j,i] * z / (one(T) - 2ϵ)
                        dxdy[k,i] += temp
                        inΔ[i,col] += outΔ[k,col] * temp
                    end
                end
            end
        end
        @inbounds sum_tmp += clamped_x
        @inbounds if proj
            unclamped_x = one(T) - sum_tmp
            clamped_x = _clamp(unclamped_x, 0, 1)
        else
            unclamped_x = one(T) - sum_tmp - y[K]
            clamped_x = _clamp(unclamped_x, 0, 1)
            if unclamped_x == clamped_x
                inΔ[K,col] += -outΔ[K,col]
            end
        end
        @inbounds if unclamped_x == clamped_x
            for i in 1:K-1
                @simd for j in i:K-1
                    inΔ[i,col] += -outΔ[K,col] * dxdy[j,i]
                end
            end
        end
    end
    return inΔ
end
=#
