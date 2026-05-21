module BijectorsReverseDiffExt

using ReverseDiff:
    ReverseDiff,
    @grad,
    value,
    track,
    TrackedReal,
    TrackedVector,
    TrackedMatrix,
    @grad_from_chainrules

using Bijectors:
    Elementwise,
    SimplexBijector,
    maphcat,
    simplex_link_jacobian,
    simplex_invlink_jacobian,
    simplex_logabsdetjac_gradient,
    Inverse
import Bijectors:
    Bijectors,
    _eps,
    logabsdetjac,
    _logabsdetjac_scale,
    _simplex_bijector,
    _simplex_inv_bijector,
    replace_diag,
    jacobian,
    _inv_link_chol_lkj,
    _link_chol_lkj,
    _transform_ordered,
    _transform_inverse_ordered,
    find_alpha,
    pd_from_lower,
    lower_triangular,
    upper_triangular,
    transpose_eager,
    cholesky_lower,
    cholesky_upper

using Bijectors.LinearAlgebra
using Bijectors.Distributions: LocationScale

_eps(::Type{<:TrackedReal{T}}) where {T} = _eps(T)
function Base.minimum(d::LocationScale{<:TrackedReal})
    m = minimum(d.ρ)
    if isfinite(m)
        return d.μ + d.σ * m
    else
        return m
    end
end
function Base.maximum(d::LocationScale{<:TrackedReal})
    m = maximum(d.ρ)
    if isfinite(m)
        return d.μ + d.σ * m
    else
        return m
    end
end

function logabsdetjac(b::Elementwise{typeof(log)}, x::Union{TrackedVector,TrackedMatrix})
    return track(logabsdetjac, b, x)
end
@grad function logabsdetjac(b::Elementwise{typeof(log)}, x::AbstractVector)
    return -sum(log, value(x)), Δ -> (nothing, -Δ ./ value(x))
end
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::Real, v::Val{0})
    return _logabsdetjac_scale(value(a), value(x), Val(0)),
    Δ -> (inv(value(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, v::Val{0})
    da = value(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, value(x), Val(0)),
    Δ -> (transpose(J) * Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, v::Val{0})
    da = value(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, value(x), Val(0)),
    Δ -> (transpose(J) * Δ, nothing, nothing)
end
# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return track(_logabsdetjac_scale, a, value(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, v::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    da = value(a)
    J = inv.(da)
    return _logabsdetjac_scale(da, value(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    return track(_logabsdetjac_scale, a, value(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, v::Val{1})
    da = value(a)
    Jᵀ = repeat(inv.(da), 1, size(x, 2))
    return _logabsdetjac_scale(da, value(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end
function _simplex_bijector(X::Union{TrackedVector,TrackedMatrix}, b::SimplexBijector)
    return track(_simplex_bijector, X, b)
end
@grad function _simplex_bijector(Y::AbstractVector, b::SimplexBijector)
    Yd = value(Y)
    return _simplex_bijector(Yd, b), Δ -> (simplex_link_jacobian(Yd)' * Δ, nothing)
end

function _simplex_inv_bijector(X::Union{TrackedVector,TrackedMatrix}, b::SimplexBijector)
    return track(_simplex_inv_bijector, X, b)
end
@grad function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector)
    Yd = value(Y)
    return _simplex_inv_bijector(Yd, b), Δ -> (simplex_invlink_jacobian(Yd)' * Δ, nothing)
end
@grad function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector)
    Yd = value(Y)
    return _simplex_inv_bijector(Yd, b),
    Δ -> begin
        maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
            simplex_invlink_jacobian(c1)' * c2
        end,
        nothing
    end
end

replace_diag(::typeof(log), X::TrackedMatrix) = track(replace_diag, log, X)
@grad function replace_diag(::typeof(log), X)
    Xd = value(X)
    f(i, j) = i == j ? log(Xd[i, j]) : Xd[i, j]
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = i == j ? ∇[i, j] / Xd[i, j] : ∇[i, j]
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

replace_diag(::typeof(exp), X::TrackedMatrix) = track(replace_diag, exp, X)
@grad function replace_diag(::typeof(exp), X)
    Xd = value(X)
    f(i, j) = ifelse(i == j, exp(Xd[i, j]), Xd[i, j])
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = ifelse(i == j, ∇[i, j] * exp(Xd[i, j]), ∇[i, j])
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

function logabsdetjac(b::SimplexBijector, x::Union{TrackedVector,TrackedMatrix})
    return track(logabsdetjac, b, x)
end
@grad function logabsdetjac(b::SimplexBijector, x::AbstractVector)
    xd = value(x)
    return logabsdetjac(b, xd), Δ -> begin
        (nothing, simplex_logabsdetjac_gradient(xd) * Δ)
    end
end

pd_from_lower(X::TrackedMatrix) = track(pd_from_lower, X)
@grad function pd_from_lower(X::AbstractMatrix)
    Xd = value(X)
    return LowerTriangular(Xd) * LowerTriangular(Xd)',
    Δ -> begin
        Xl = LowerTriangular(Xd)
        return (LowerTriangular(Δ' * Xl + Δ * Xl),)
    end
end

@grad_from_chainrules pd_from_upper(X::TrackedMatrix)

lower_triangular(A::TrackedMatrix) = track(lower_triangular, A)
@grad function lower_triangular(A::AbstractMatrix)
    Ad = value(A)
    return lower_triangular(Ad), Δ -> (lower_triangular(Δ),)
end

upper_triangular(A::TrackedMatrix) = track(upper_triangular, A)
@grad function upper_triangular(A::AbstractMatrix)
    Ad = value(A)
    return upper_triangular(Ad), Δ -> (upper_triangular(Δ),)
end

function find_alpha(wt_y::T, wt_u_hat::T, b::T) where {T<:TrackedReal}
    return track(find_alpha, wt_y, wt_u_hat, b)
end
@grad function find_alpha(wt_y::TrackedReal, wt_u_hat::TrackedReal, b::TrackedReal)
    α = find_alpha(value(wt_y), value(wt_u_hat), value(b))

    ∂wt_y = inv(1 + wt_u_hat * sech(α + b)^2)
    ∂wt_u_hat = -tanh(α + b) * ∂wt_y
    ∂b = ∂wt_y - 1
    find_alpha_pullback(Δ::Real) = (Δ * ∂wt_y, Δ * ∂wt_u_hat, Δ * ∂b)

    return α, find_alpha_pullback
end

# `OrderedBijector`
@grad_from_chainrules _transform_ordered(y::Union{TrackedVector,TrackedMatrix})
@grad_from_chainrules _transform_inverse_ordered(x::Union{TrackedVector,TrackedMatrix})

@grad_from_chainrules Bijectors.update_triu_from_vec(
    vals::TrackedVector{<:Real}, k::Int, dim::Int
)

@grad_from_chainrules _link_chol_lkj(x::TrackedMatrix)
@grad_from_chainrules _link_chol_lkj_from_upper(x::TrackedMatrix)
@grad_from_chainrules _link_chol_lkj_from_lower(x::TrackedMatrix)

cholesky_lower(X::TrackedMatrix) = track(cholesky_lower, X)
cholesky_upper(X::TrackedMatrix) = track(cholesky_upper, X)

# `cholesky_lower` and `cholesky_upper` route through ChainRules.jl rules for
# `Hermitian` and `cholesky` (see `BijectorsReverseDiffChainRulesExt`). Without
# ChainRules loaded that extension is dormant and the user hits a `MethodError` on
# `ReverseDiff.track(cholesky_lower, ::TrackedMatrix)`. Point them at the fix.
const _CHAINRULES_GATED = (cholesky_lower, cholesky_upper)

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _kwargs
            exc.f === ReverseDiff.track || return nothing
            length(argtypes) == 2 || return nothing
            T = argtypes[1]
            # `.instance` is only defined for singleton types (e.g. `typeof(some_fn)`).
            isdefined(T, :instance) || return nothing
            T.instance in _CHAINRULES_GATED || return nothing
            return print(
                io,
                "\nDifferentiating `$(nameof(T.instance))` with ReverseDiff requires ChainRules.jl. ",
                "Run `using ChainRules` first.",
            )
        end
    end
end

transpose_eager(X::TrackedMatrix) = track(transpose_eager, X)
@grad function transpose_eager(X_tracked::TrackedMatrix)
    X = value(X_tracked)
    transpose_eager_pullback(Δ) = (transpose_eager(Δ),)
    return transpose_eager(X), transpose_eager_pullback
end

end
