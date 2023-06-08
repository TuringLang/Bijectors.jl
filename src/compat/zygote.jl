using .Zygote: Zygote, @adjoint, pullback

using Compat: eachcol

@adjoint istraining() = true, _ -> nothing

@adjoint function mapvcat(f, args...)
    g(f, args...) = map(f, args...)
    return pullback(g, f, args...)
end
@adjoint function eachcolmaphcat(f, x1, x2)
    function g(f, x1, x2)
        init = reshape(f(view(x1, :, 1), x2[1]), :, 1)
        return reduce(hcat, [f(view(x1, :, i), x2[i]) for i in 2:size(x1, 2)]; init=init)
    end
    return pullback(g, f, x1, x2)
end
@adjoint function eachcolmaphcat(f, x)
    function g(f, x)
        init = reshape(f(view(x, :, 1)), :, 1)
        return reduce(hcat, [f(view(x, :, i)) for i in 2:size(x, 2)]; init=init)
    end
    return pullback(g, f, x)
end
@adjoint function sumeachcol(f, x1, x2)
    g(f, x1, x2) = sum([f(view(x1, :, i), x2[i]) for i in 1:size(x1, 2)])
    return pullback(g, f, x1, x2)
end

@adjoint function logabsdetjac(b::Elementwise{typeof(log)}, x::AbstractVector)
    return -sum(log, x), Δ -> (nothing, -Δ ./ x)
end

# AD implementations
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
## Positive definite matrices
@adjoint function replace_diag(::typeof(log), X)
    f(i, j) = i == j ? log(X[i, j]) : X[i, j]
    out = f.(1:size(X, 1), (1:size(X, 2))')
    return out, ∇ -> begin
        g(i, j) = i == j ? ∇[i, j] / X[i, j] : ∇[i, j]
        (nothing, g.(1:size(X, 1), (1:size(X, 2))'))
    end
end
@adjoint function replace_diag(::typeof(exp), X)
    f(i, j) = ifelse(i == j, exp(X[i, j]), X[i, j])
    out = f.(1:size(X, 1), (1:size(X, 2))')
    return out, ∇ -> begin
        g(i, j) = ifelse(i == j, ∇[i, j] * exp(X[i, j]), ∇[i, j])
        (nothing, g.(1:size(X, 1), (1:size(X, 2))'))
    end
end

@adjoint function pd_logpdf_with_trans(d, X::AbstractMatrix{<:Real}, transform::Bool)
    return pullback(pd_logpdf_with_trans_zygote, d, X, transform)
end
function pd_logpdf_with_trans_zygote(d, X::AbstractMatrix{<:Real}, transform::Bool)
    T = eltype(X)
    Xcf = cholesky(X; check=false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I; check=true)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        factors = Xcf.factors
        n = size(d, 1)
        k = n + 2
        @inbounds for i in diagind(factors)
            k -= 1
            lp += k * log(factors[i])
        end
        lp += n * oftype(lp, IrrationalConstants.logtwo)
    end
    return lp
end

# Simplex adjoints

@adjoint function _simplex_bijector(X::AbstractVector, b::SimplexBijector)
    return _simplex_bijector(X, b), Δ -> (simplex_link_jacobian(X)' * Δ, nothing)
end
@adjoint function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector)
    return _simplex_inv_bijector(Y, b), Δ -> (simplex_invlink_jacobian(Y)' * Δ, nothing)
end

@adjoint function _simplex_bijector(X::AbstractMatrix, b::SimplexBijector)
    return _simplex_bijector(X, b),
    Δ -> begin
        maphcat(eachcol(X), eachcol(Δ)) do c1, c2
            simplex_link_jacobian(c1)' * c2
        end,
        nothing
    end
end
@adjoint function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector)
    return _simplex_inv_bijector(Y, b),
    Δ -> begin
        maphcat(eachcol(Y), eachcol(Δ)) do c1, c2
            simplex_invlink_jacobian(c1)' * c2
        end,
        nothing
    end
end

@adjoint function logabsdetjac(b::SimplexBijector, x::AbstractVector)
    return logabsdetjac(b, x), Δ -> begin
        (nothing, simplex_logabsdetjac_gradient(x) * Δ)
    end
end

# LocationScale fix

@adjoint function minimum(d::LocationScale)
    function _minimum(d)
        m = minimum(d.ρ)
        if isfinite(m)
            return d.μ + d.σ * m
        else
            return m
        end
    end
    return pullback(_minimum, d)
end
@adjoint function maximum(d::LocationScale)
    function _maximum(d)
        m = maximum(d.ρ)
        if isfinite(m)
            return d.μ + d.σ * m
        else
            return m
        end
    end
    return pullback(_maximum, d)
end
@adjoint function lower_triangular(A::AbstractMatrix)
    return lower_triangular(A), Δ -> (lower_triangular(Δ),)
end
@adjoint function pd_from_lower(X::AbstractMatrix)
    return LowerTriangular(X) * LowerTriangular(X)',
    Δ -> begin
        Xl = LowerTriangular(X)
        return (LowerTriangular(Δ' * Xl + Δ * Xl),)
    end
end
@adjoint function pd_link(X::AbstractMatrix{<:Real})
    return pullback(X) do X
        Y = cholesky(X; check=true).L
        return replace_diag(log, Y)
    end
end
