using .Zygote: Zygote, @adjoint, @nograd, pullback

@adjoint istraining() = true, _ -> nothing
@nograd Bijectors._debug

@adjoint function mapvcat(f, args...)
    g(f, args...) = map(f, args...)
    return pullback(g, f, args...)
end
@adjoint function eachcolmaphcat(f, x1, x2)
    function g(f, x1, x2)
        init = reshape(f(view(x1, :, 1), x2[1]), :, 1)
        return reduce(hcat, [f(view(x1, :, i), x2[i]) for i in 2:size(x1, 2)]; init = init)
    end
    return pullback(g, f, x1, x2)
end
@adjoint function eachcolmaphcat(f, x)
    function g(f, x)
        init = reshape(f(view(x, :, 1)), :, 1)
        return reduce(hcat, [f(view(x, :, i)) for i in 2:size(x, 2)]; init = init)
    end
    return pullback(g, f, x)
end
@adjoint function _sum(f, args...)
    g(f, args...) = sum(map(f, args...))
    return pullback(g, f, args...)
end
@adjoint function _sumeachcol(f, x1, x2)
    g(f, x1, x2) = sum([f(view(x1, :, i), x2[i]) for i in 1:size(x1, 2)])
    return pullback(g, f, x1, x2)
end

@adjoint function logabsdetjac(b::Log{1}, x::AbstractVector)
    return -sum(log, x), Δ -> (nothing, -Δ ./ x)
end
@adjoint function logabsdetjac(b::Log{1}, x::AbstractMatrix)
    return -vec(sum(log, x; dims = 1)), Δ -> (nothing, .- Δ' ./ x)
end

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:ZygoteAD}, Inverse{<:ADBijector{<:ZygoteAD}}},
    x::Real
)
    return Zygote.gradient(b, x)[1]
end
function jacobian(
    b::Union{<:ADBijector{<:ZygoteAD}, Inverse{<:ADBijector{<:ZygoteAD}}},
    x::AbstractVector{<:Real}
)
    return Zygote.jacobian(b, x)
end
@adjoint function _logabsdetjac_scale(a::Real, x::Real, ::Val{0})
    return _logabsdetjac_scale(a, x, Val(0)), Δ -> (inv(a) .* Δ, nothing, nothing)
end
@adjoint function _logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0})
    J = fill(inv.(a), length(x))
    return _logabsdetjac_scale(a, x, Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
@adjoint function _logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{0})
    J = fill(size(x, 1) / a, size(x, 2))
    return _logabsdetjac_scale(a, x, Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
@adjoint function _logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    J = inv.(a)
    return _logabsdetjac_scale(a, x, Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
@adjoint function _logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1})
    Jᵀ = repeat(inv.(a), 1, size(x, 2))
    return _logabsdetjac_scale(a, x, Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end

@adjoint function (b::PDBijector)(X::AbstractMatrix{<:Real})
    function f(X::AbstractMatrix{<:Real})
        Y = Zygote.Buffer(X)
        Y .= cholesky(X).L
        @inbounds for i in diagind(X)
            Y[i] = log(Y[i])
        end
        return copy(Y)
    end
    return pullback(f, X)
end

@adjoint function (ib::Inverse{PDBijector})(Y::AbstractMatrix{<:Real})
    function f(Y::AbstractMatrix{<:Real})
        X = Zygote.Buffer(Y)
        X .= Y
        @inbounds for i in diagind(Y)
            X[i] = exp(X[i])
        end
        _X = copy(X)
        return LowerTriangular(_X) * LowerTriangular(_X)'
    end
    return pullback(f, Y)
end

@adjoint function _logpdf_with_trans_pd(
    d,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    return pullback(_logpdf_with_trans_pd_zygote, d, X, transform)
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
@adjoint function unsafe_cholesky(Σ::Real, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Σ), nothing)
        (Δ.factors[1, 1] / (2 * C.U[1, 1]), nothing)
    end
end
@adjoint function unsafe_cholesky(Σ::Diagonal, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || (Diagonal(zero(diag(Δ.factors))), nothing)
        (Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)), nothing)
    end
end
@adjoint function unsafe_cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}}, check)
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

# Simplex adjoints

@adjoint function _simplex_bijector(X::AbstractVector, b::SimplexBijector)
    return _simplex_bijector(X, b), Δ -> (simplex_link_jacobian(X)' * Δ, nothing)
end
@adjoint function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector)
    return _simplex_inv_bijector(Y, b), Δ -> (simplex_invlink_jacobian(Y)' * Δ, nothing)
end

@adjoint function _simplex_bijector(X::AbstractMatrix, b::SimplexBijector)
    return _simplex_bijector(X, b), Δ -> begin
        maphcat(eachcol(X), eachcol(Δ)) do c1, c2
            simplex_link_jacobian(c1)' * c2
        end, nothing
    end
end
@adjoint function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector)
    return _simplex_inv_bijector(Y, b), Δ -> begin
        maphcat(eachcol(Y), eachcol(Δ)) do c1, c2
            simplex_invlink_jacobian(c1)' * c2
        end, nothing
    end
end

@adjoint function logabsdetjac(b::SimplexBijector, x::AbstractVector)
    return logabsdetjac(b, x), Δ -> begin
        (nothing, simplex_logabsdetjac_gradient(x) * Δ)
    end
end
@adjoint function logabsdetjac(b::SimplexBijector, x::AbstractMatrix)
    return logabsdetjac(b, x), Δ -> begin
        (nothing, maphcat(eachcol(x), Δ) do c, g
            simplex_logabsdetjac_gradient(c) * g
        end)
    end
end

# LocationScale fix

@adjoint function _clamp(x::T, dist::LocationScale) where {T <: Real}
    function f(x, dist)
        ϵ = _eps(T)
        bounds = (minimum(dist) + ϵ, maximum(dist) - ϵ)
        if x < bounds[1]
            clamped_x = bounds[1]
        elseif x > bounds[2]
            clamped_x = bounds[2]
        else
            clamped_x = x
        end
        DEBUG && _debug("x = $x, bounds = $bounds, clamped_x = $clamped_x")
        return clamped_x
    end
    return pullback(f, x, dist)
end
