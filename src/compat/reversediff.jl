module ReverseDiffCompat

using ..ReverseDiff: ReverseDiff, @grad, value, track, TrackedReal, TrackedVector, 
    TrackedMatrix
using Requires, LinearAlgebra

using ..Bijectors: Log, SimplexBijector, maphcat, simplex_link_jacobian, 
    simplex_invlink_jacobian, simplex_logabsdetjac_gradient, ADBijector, 
    ReverseDiffAD, Inverse
import ..Bijectors: _eps, logabsdetjac, _logabsdetjac_scale, _simplex_bijector, 
    _simplex_inv_bijector, replace_diag, jacobian, getpd, lower, 
    _inv_link_chol_lkj, _link_chol_lkj

using Compat: eachcol
using Distributions: LocationScale

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:ReverseDiffAD}, Inverse{<:ADBijector{<:ReverseDiffAD}}},
    x::Real
)
    return ReverseDiff.gradient(x -> b(x[1]), [x])[1]
end
function jacobian(
    b::Union{<:ADBijector{<:ReverseDiffAD}, Inverse{<:ADBijector{<:ReverseDiffAD}}},
    x::AbstractVector{<:Real}
)
    return ReverseDiff.jacobian(b, x)
end

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

logabsdetjac(b::Log{1}, x::Union{TrackedVector, TrackedMatrix}) = track(logabsdetjac, b, x)
@grad function logabsdetjac(b::Log{1}, x::AbstractVector)
    return -sum(log, value(x)), Δ -> (nothing, -Δ ./ value(x))
end
@grad function logabsdetjac(b::Log{1}, x::AbstractMatrix)
    return -vec(sum(log, value(x); dims = 1)), Δ -> (nothing, .- Δ' ./ value(x))
end
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::Real, v::Val{0})
    return _logabsdetjac_scale(value(a), value(x), Val(0)), Δ -> (inv(value(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, v::Val{0})
    da = value(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, value(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(_logabsdetjac_scale, a, value(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, v::Val{0})
    da = value(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, value(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
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
function _simplex_bijector(X::Union{TrackedVector, TrackedMatrix}, b::SimplexBijector{1})
    return track(_simplex_bijector, X, b)
end
@grad function _simplex_bijector(Y::AbstractVector, b::SimplexBijector{1})
    Yd = value(Y)
    return _simplex_bijector(Yd, b), Δ -> (simplex_link_jacobian(Yd)' * Δ, nothing)
end
@grad function _simplex_bijector(Y::AbstractMatrix, b::SimplexBijector{1})
    Yd = value(Y)
    return _simplex_bijector(Yd, b), Δ -> begin
        maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
            simplex_link_jacobian(c1)' * c2
        end, nothing
    end
end

function _simplex_inv_bijector(X::Union{TrackedVector, TrackedMatrix}, b::SimplexBijector{1})
    return track(_simplex_inv_bijector, X, b)
end
@grad function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector{1})
    Yd = value(Y)
    return _simplex_inv_bijector(Yd, b), Δ -> (simplex_invlink_jacobian(Yd)' * Δ, nothing)
end
@grad function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector{1})
    Yd = value(Y)
    return _simplex_inv_bijector(Yd, b), Δ -> begin
        maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
            simplex_invlink_jacobian(c1)' * c2
        end, nothing
    end
end

replace_diag(::typeof(log), X::TrackedMatrix) = track(replace_diag, log, X)
@grad function replace_diag(::typeof(log), X)
    Xd = value(X)
    f(i, j) = i == j ? log(Xd[i, j]) : Xd[i, j]
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = i == j ? ∇[i, j]/Xd[i, j] : ∇[i, j]
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

replace_diag(::typeof(exp), X::TrackedMatrix) = track(replace_diag, exp, X)
@grad function replace_diag(::typeof(exp), X)
    Xd = value(X)
    f(i, j) = ifelse(i == j, exp(Xd[i, j]), Xd[i, j])
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = ifelse(i == j, ∇[i, j]*exp(Xd[i, j]), ∇[i, j])
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

logabsdetjac(b::SimplexBijector{1}, x::Union{TrackedVector, TrackedMatrix}) = track(logabsdetjac, b, x)
@grad function logabsdetjac(b::SimplexBijector{1}, x::AbstractVector)
    xd = value(x)
    return logabsdetjac(b, xd), Δ -> begin
        (nothing, simplex_logabsdetjac_gradient(xd) * Δ)
    end
end
@grad function logabsdetjac(b::SimplexBijector{1}, x::AbstractMatrix)
    xd = value(x)
    return logabsdetjac(b, xd), Δ -> begin
        (nothing, maphcat(eachcol(xd), Δ) do c, g
            simplex_logabsdetjac_gradient(c) * g
        end)
    end
end

getpd(X::TrackedMatrix) = track(getpd, X)
@grad function getpd(X::AbstractMatrix)
    Xd = value(X)
    return LowerTriangular(Xd) * LowerTriangular(Xd)', Δ -> begin
        Xl = LowerTriangular(Xd)
        return (LowerTriangular(Δ' * Xl + Δ * Xl),)
    end
end
lower(A::TrackedMatrix) = track(lower, A)
@grad function lower(A::AbstractMatrix)
    Ad = value(A)
    return lower(Ad), Δ -> (lower(Δ),)
end


_inv_link_chol_lkj(y::TrackedMatrix) = track(_inv_link_chol_lkj, y)
@grad function _inv_link_chol_lkj(y_tracked)
    y = value(y_tracked)

    LinearAlgebra.checksquare(y)

    K = size(y, 1)

    z = tanh.(y)
    w = similar(z)

    @inbounds for j in 1:K
        w[1, j] = 1
        for i in j+1:K
            w[i, j] = 0
        end
        for i in 2:j
            w[i, j] = w[i-1, j] * sqrt(1 - z[i-1, j]^2)
        end
    end

    w1 = copy(w) # cache result

    @inbounds for j in 2:K
        for i in 1:j-1
            w[i, j] = w[i, j] * z[i, j]
        end
    end

    function pullback_inv_link_chol_lkj(Δw)
        LinearAlgebra.checksquare(Δw)

        Δz = zero(Δw)
        Δw1 = zero(Δw)
        @inbounds for j=2:K, i=1:j-1
            Δw1[i,j] = Δw[i,j] * z[i,j]
            Δz[i,j] = Δw[i,j] * w1[i,j]
        end
        @inbounds for i in 1:K
            Δw1[i,i] = Δw[i,i]
        end

        @inbounds for j=2:K, i=j:-1:2
            tz = sqrt(1 - z[i-1, j]^2)
            Δw1[i-1, j] += Δw1[i, j] * tz
            Δz[i-1, j] += Δw1[i, j] * w1[i-1, j] * 0.5 / tz * (-2 * z[i-1, j])
        end

        Δy = Δz .* (1 ./ cosh.(y).^2)
        return (Δy,)
    end

    return w, pullback_inv_link_chol_lkj
end

# zero(x) is used to convert w to dense matrix, because ReverseDiff doesn't support track
# LowerTriangular matrix (concretely, Base.IndexStyle(::A) != Base.IndexLinear())
# See: http://www.juliadiff.org/ReverseDiff.jl/limits/
_link_chol_lkj(w::TrackedMatrix) = track(_link_chol_lkj, w + zero(x))
@grad function _link_chol_lkj(w_tracked)
    w = value(w_tracked)

    LinearAlgebra.checksquare(w)

    K = size(w, 1)
    z = similar(w)

    # This block can't be integrated with loop below, because w[1,1] != 0.
    @inbounds for i=1:K
        z[i, 1] = 0
    end

    @inbounds for j=2:K
        z[1, j] = w[1, j]
        for i=j:K
            z[i, j] = 0
        end
        for i=2:j-1
            p = w[i, j]
            for ip in 1:(i-1)
                p /= sqrt(1-z[ip, j]^2)
            end
            z[i, j] = p
        end
    end
    
    y = atanh.(z)

    function pull_link_chol_lkj(Δy)
        LinearAlgebra.checksquare(Δy)

        zt0 = 1 ./ (1 .- z.^2)
        zt = sqrt.(zt0)
        Δz = Δy .* zt0
        Δw = zero(Δy)
        
        @inbounds for j=2:K, i=(j-1):-1:2
            pd = prod(zt[1:i-1,j])
            Δw[i,j] += Δz[i,j] * pd
            for ip in 1:(i-1)
                Δw[ip, j] += Δz[i,j] * w[i,j] * pd / (1-z[ip,j]^2) * z[ip,j]
            end
        end
        @inbounds for j=2:K
            Δw[1, j] += Δz[1, j]
        end

        return (Δw,)
    end

    return y, pull_link_chol_lkj
    
end

end
