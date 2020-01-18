Zygote.@nograd Bijectors._debug

Zygote.@adjoint function _bcall(
    f::SimplexBijector{proj},
    x::AbstractVecOrMat{<:Real},
) where {proj}
    val, back = Zygote.pullback(
        x -> copy(_simplex_bijector_nosimd!(Zygote.Buffer(x), x, Val(proj))),
        x,
    )
    val, x -> (nothing, back(x)...,)
end
Zygote.@adjoint function _bcall(
    f::Inversed{<:SimplexBijector{proj}},
    y::AbstractVecOrMat{<:Real},
) where {proj}
    val, back = Zygote.pullback(
        y -> copy(_simplex_inv_bijector_nosimd!(Zygote.Buffer(y), y, f, Val(proj))),
        y,
    )
    val, x -> (nothing, back(x)...,)
end

# Zygote doesn't support SIMD
function _simplex_bijector_nosimd!(
    y,
    x::AbstractVector{T},
    ::Val{proj},
) where {T, proj}
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    sum_tmp = zero(T)
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = StatsFuns.logit(z) + log(T(K - 1))
    @inbounds for k in 2:(K - 1)
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
function _simplex_bijector_nosimd!(
    Y,
    X::AbstractMatrix{T},
    ::Val{proj},
) where {T, proj}
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds for n in 1:size(X, 2)
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
function _simplex_inv_bijector_nosimd!(
    x,
    y::AbstractVector{T},
    ib,
    ::Val{proj},
) where {T, proj}
    K = length(y)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), ib.orig)
    sum_tmp = zero(T)
    @inbounds for k = 2:(K - 1)
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
function _simplex_inv_bijector_nosimd!(
    X,
    Y::AbstractMatrix{T},
    ib,
    ::Val{proj},
) where {T, proj}
    K, N = size(Y, 1), size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"

    ϵ = _eps(T)
    @inbounds for n in 1:size(X, 2)
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

Zygote.@adjoint function logabsdetjac(b::SimplexBijector, x::AbstractVector)
    val, back = Zygote.pullback(_logabsdetjac_nosimd, b, x)
    val, x -> back(x)
end
function _logabsdetjac_nosimd(b::SimplexBijector, x::AbstractVector{T}) where {T}
    ϵ = _eps(T)
    lp = zero(T)
    
    K = length(x)

    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    lp += log(z + ϵ) + log((one(T) + ϵ) - z)
    @inbounds for k in 2:(K - 1)
        sum_tmp += x[k-1]
        z = x[k] / ((one(T) + ϵ) - sum_tmp)
        lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
    end

    return -lp
end

Zygote.@adjoint function _link_pd(
    d,
    X::AbstractMatrix{<:Real},
)
    val, back = Zygote.pullback(_link_pd_zygote, d, X)
    val, x -> back(x)
end
function _link_pd_zygote(
    d,
    X::AbstractMatrix{T},
) where {T <: Real}
    Y = Zygote.Buffer(X)
    Y .= cholesky(X).L
    @inbounds for i in diagind(X)
        Y[i] = log(Y[i])
    end
    return copy(Y)
end

Zygote.@adjoint function _invlink_pd(
    d,
    X::AbstractMatrix{<:Real},
)
    val, back = Zygote.pullback(_invlink_pd_zygote, d, X)
    val, x -> back(x)
end
function _invlink_pd_zygote(
    d,
    Y::AbstractMatrix{T},
) where {T <: Real}
    X = Zygote.Buffer(Y)
    X .= Y
    @inbounds for i in diagind(Y)
        X[i] = exp(X[i])
    end
    _X = copy(X)
    return LowerTriangular(_X) * LowerTriangular(_X)'
end

Zygote.@adjoint function _logpdf_with_trans_pd(
    d,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    val, back = Zygote.pullback(_logpdf_with_trans_pd_zygote, d, X, transform)
    val, x -> back(x)
end
function _logpdf_with_trans_pd_zygote(
    d,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    T = eltype(X)
    Xcf = unsafe_cholesky(X, false)
    if !issuccess(Xcf)
        Xcf = unsafe_cholesky(X + (eps(T) * norm(X)) * I, true)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        U = Xcf.U
        @inbounds for i in 1:dim(d)
            lp += (dim(d) - i + 2) * log(U[i, i])
        end
        lp += dim(d) * log(T(2))
    end
    return lp
end

# Zygote doesn't support kwargs, e.g. cholesky(A, check = false), hence this workaround
# Copied from DistributionsAD
unsafe_cholesky(x, check) = cholesky(x, check=check)
Zygote.@adjoint function unsafe_cholesky(Σ::Real, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Σ), nothing)
        (Δ.factors[1, 1] / (2 * C.U[1, 1]), nothing)
    end
end
Zygote.@adjoint function unsafe_cholesky(Σ::Diagonal, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || (Diagonal(zero(diag(Δ.factors))), nothing)
        (Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)), nothing)
    end
end
Zygote.@adjoint function unsafe_cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}}, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Δ.factors), nothing)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = LinearAlgebra.copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄), nothing)
    end
end
