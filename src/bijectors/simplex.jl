####################
# Simplex bijector #
####################
struct SimplexBijector <: Bijector end

output_size(::SimplexBijector, sz::Tuple{Int}) = (first(sz) - 1,)
output_size(::Inverse{SimplexBijector}, sz::Tuple{Int}) = (first(sz) + 1,)

output_size(::SimplexBijector, sz::Tuple{Int,Int}) = Base.setindex(sz, first(sz) - 1, 1)
output_size(::Inverse{SimplexBijector}, sz::Tuple{Int,Int}) = Base.setindex(sz, first(sz) + 1, 1)

with_logabsdet_jacobian(b::SimplexBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(b::SimplexBijector, x) = _simplex_bijector(x, b)
transform!(b::SimplexBijector, y, x) = _simplex_bijector!(y, x, b)

function _simplex_bijector(x::AbstractArray, b::SimplexBijector)
    sz = size(x)
    K = size(x, 1)
    y = similar(x, Base.setindex(sz, K - 1, 1))
    _simplex_bijector!(y, x, b)
    return y
end

# Vector implementation.
function _simplex_bijector!(y, x::AbstractVector, ::SimplexBijector)
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(x)
    ϵ = _eps(T)
    sum_tmp = zero(T)
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = LogExpFunctions.logit(z) + log(T(K - 1))
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        y[k] = LogExpFunctions.logit(z) + log(T(K - k))
    end
    return y
end

# Matrix implementation.
function _simplex_bijector!(Y, X::AbstractMatrix, ::SimplexBijector)
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(X)
    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp = zero(T)
        z = X[1, n] * (one(T) - 2ϵ) + ϵ
        Y[1, n] = LogExpFunctions.logit(z) + log(T(K - 1))
        for k in 2:(K - 1)
            sum_tmp += X[k - 1, n]
            z = (X[k, n] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
            Y[k, n] = LogExpFunctions.logit(z) + log(T(K - k))
        end
    end

    return Y
end

# Inverse.
function transform(ib::Inverse{<:SimplexBijector}, y::AbstractArray)
    return _simplex_inv_bijector(y, ib.orig)
end
function transform!(
    ib::Inverse{<:SimplexBijector}, x::AbstractArray{T}, y::AbstractArray{T}
) where {T}
    return _simplex_inv_bijector!(x, y, ib.orig)
end

function _simplex_inv_bijector(y, b)
    sz = size(y)
    K = sz[1] + 1
    x = similar(y, Base.setindex(sz, K, 1))
    _simplex_inv_bijector!(x, y, b)
    return x
end

function _simplex_inv_bijector!(x, y::AbstractVector, b::SimplexBijector)
    K = length(y) + 1
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(y)
    ϵ = _eps(T)
    @inbounds z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
    sum_tmp = zero(T)
    @inbounds @simd for k in 2:(K - 1)
        z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k - 1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
    end
    @inbounds sum_tmp += x[K - 1]
    x[K] = _clamp(one(T) - sum_tmp, 0, 1)
    return x
end

function _simplex_inv_bijector!(X, Y::AbstractMatrix, b::SimplexBijector)
    K, N = size(Y, 1) + 1, size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(Y)
    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp, z = zero(T), LogExpFunctions.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
        for k in 2:(K - 1)
            z = LogExpFunctions.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
        end
        sum_tmp += X[K - 1, n]
        X[K, n] = _clamp(one(T) - sum_tmp, 0, 1)
    end

    return X
end

function logabsdetjac(b::SimplexBijector, x::AbstractVector{T}) where {T}
    ϵ = _eps(T)
    lp = zero(T)

    K = length(x)

    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ))
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        z = x[k] / max(one(T) - sum_tmp, ϵ)
        lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ)) + log(max(one(T) - sum_tmp, ϵ))
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
    c1 = z >= ϵ
    zc = one(T) - z
    c2 = zc >= ϵ
    g[1] = ifelse(c1 & c2, -1 / z + 1 / zc, ifelse(c1, -1 / z, 1 / zc))
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        temp = 1 / (1 - sum_tmp)
        c0 = temp >= ϵ
        z = ifelse(c0, x[k] * temp, x[k] / ϵ)
        #lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log(temp)
        dzdx = ifelse(c0, temp, one(T))
        c1 = z >= ϵ
        zc = one(T) - z
        c2 = zc >= ϵ
        dldz = ifelse(c1 & c2, 1 / z - 1 / zc, ifelse(c1, 1 / z, -1 / zc))
        dldx = dldz * dzdx
        g[k] -= dldx
        for i in 1:(k - 1)
            dzdxp = ifelse(c0, x[k] * dzdx^2, zero(T))
            dldxp = dldz * dzdxp - ifelse(c0, temp, zero(T))
            g[i] -= dldxp
        end
    end
    return g
end

function simplex_logabsdetjac_gradient(x::AbstractMatrix)
    T = eltype(x)
    ϵ = _eps(T)
    K = size(x, 1)
    g = similar(x)
    g .= 0
    @inbounds @simd for col in 1:size(x, 2)
        sum_tmp = zero(eltype(x))
        z = x[1, col]
        #lp += log(z + ϵ) + log((one(T) + ϵ) - z)
        c1 = z >= ϵ
        zc = one(T) - z
        c2 = zc >= ϵ
        g[1, col] = ifelse(c1 & c2, -1 / z + 1 / zc, ifelse(c1, -1 / z, 1 / zc))
        for k in 2:(K - 1)
            sum_tmp += x[k - 1, col]
            temp = 1 / (1 - sum_tmp)
            c0 = temp >= ϵ
            z = ifelse(c0, x[k, col] * temp, x[k, col] / ϵ)
            #lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log(temp)
            dzdx = ifelse(c0, temp, one(T))
            c1 = z >= ϵ
            zc = one(T) - z
            c2 = zc >= ϵ
            dldz = ifelse(c1 & c2, 1 / z - 1 / zc, ifelse(c1, 1 / z, -1 / zc))
            dldx = dldz * dzdx
            g[k, col] -= dldx
            for i in 1:(k - 1)
                dzdxp = ifelse(c0, x[k, col] * dzdx^2, zero(T))
                dldxp = dldz * dzdxp - ifelse(c0, temp, zero(T))
                g[i, col] -= dldxp
            end
        end
    end
    return g
end

function simplex_link_jacobian(x::AbstractVector{T}) where {T<:Real}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    dydxt = fill!(similar(x, K, K - 1), 0)
    ϵ = _eps(T)
    sum_tmp = zero(T)

    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds dydxt[1, 1] = (1 / z + 1 / (1 - z)) * (one(T) - 2ϵ)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k - 1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        dydxt[k, k] = (1 / z + 1 / (1 - z)) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        for i in 1:(k - 1)
            dydxt[i, k] =
                (1 / z + 1 / (1 - z)) * (x[k] + ϵ) * (one(T) - 2ϵ) /
                ((one(T) + ϵ) - sum_tmp)^2
        end
    end
    return dydxt'
end
function jacobian(b::SimplexBijector, x::AbstractVector{T}) where {T}
    return simplex_link_jacobian(x)
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

function simplex_invlink_jacobian(y::AbstractVector{T}) where {T<:Real}
    K = length(y) + 1
    @assert K > 1 "x needs to be of length greater than 1"
    dxdy = fill!(similar(y, K, K - 1), 0)

    ϵ = _eps(T)
    @inbounds z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, 0, 1)
    @inbounds if unclamped_x == clamped_x
        dxdy[1, 1] = z * (1 - z) / (one(T) - 2ϵ)
    end
    sum_tmp = zero(T)
    @inbounds for k in 2:(K - 1)
        z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
        sum_tmp += clamped_x
        unclamped_x = ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ
        clamped_x = _clamp(unclamped_x, 0, 1)
        if unclamped_x == clamped_x
            dxdy[k, k] = z * (1 - z) * ((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ)
            for i in 1:(k - 1)
                for j in i:(k - 1)
                    dxdy[k, i] += -dxdy[j, i] * z / (one(T) - 2ϵ)
                end
            end
        end
    end
    @inbounds sum_tmp += clamped_x
    unclamped_x = one(T) - sum_tmp
    clamped_x = _clamp(unclamped_x, 0, 1)
    @inbounds if unclamped_x == clamped_x
        for i in 1:(K - 1)
            @simd for j in i:(K - 1)
                dxdy[K, i] += -dxdy[j, i]
            end
        end
    end
    return dxdy
end
# jacobian
function jacobian(ib::Inverse{<:SimplexBijector}, y::AbstractVector{T}) where {T}
    return simplex_invlink_jacobian(y)
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
    @inbounds z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
    unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
    clamped_x = _clamp(unclamped_x, 0, 1)
    @inbounds if unclamped_x == clamped_x
        dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
        inΔ[1] += outΔ[1] * dxdy[1,1]
    end
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
        z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
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
        z = LogExpFunction.logistic(y[1,col] - log(T(K - 1)))
        unclamped_x = (z - ϵ) / (one(T) - 2ϵ)
        clamped_x = _clamp(unclamped_x, 0, 1)
        if unclamped_x == clamped_x
            dxdy[1,1] = z * (1 - z) / (one(T) - 2ϵ)
            inΔ[1,col] += outΔ[1,col] * dxdy[1,1]
        end
        sum_tmp = zero(T)
        for k = 2:(K - 1)
            z = LogExpFunctions.logistic(y[k,col] - log(T(K - k)))
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
