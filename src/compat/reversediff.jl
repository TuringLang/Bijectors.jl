module ReverseDiffCompat

using ..ReverseDiff: ReverseDiff, value, deriv, track, SpecialInstruction
using Requires, LinearAlgebra
const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
const RTA = ReverseDiff.TrackedArray

import Base: vcat, hcat
using ..Bijectors: Log, SimplexBijector, maphcat, simplex_link_jacobian, 
    simplex_invlink_jacobian, simplex_logabsdetjac_gradient
import ..Bijectors: _eps, logabsdetjac, _logabsdetjac_scale, _simplex_bijector, 
    _simplex_inv_bijector, replace_diag

_eps(::Type{<:RTR{T}}) where {T} = _eps(T)

expr = Expr(:block)
combs = [
    [],
    [:AbstractArray],
    [:RTA],
    [:RTR],
    [:Number],
    [:AbstractArray, :RTA],
    [:AbstractArray, :RTR],
    [:AbstractArray, :Number],
    [:RTA, :RTR],
    [:RTA, :Number],
]
for c in combs, f = [:hcat, :vcat]
    cnames = map(_ -> gensym(), c)
    push!(expr.args, :(Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::Union{RTA, RTR}, xs::Union{AbstractArray, Number}...) = track($f, $(cnames...), x, xs...)))
end
using DistributionsAD
#=@init @require DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"=# @eval begin
    $expr
    DistributionsAD.ReverseDiffX.@grad function vcat(x::Real)
        vcat(value(x)), (Δ) -> (Δ[1],)
    end
    DistributionsAD.ReverseDiffX.@grad function vcat(x1::Real, x2::Real)
        vcat(value(x1), value(x2)), (Δ) -> (Δ[1], Δ[2])
    end
    DistributionsAD.ReverseDiffX.@grad function vcat(x1::AbstractVector, x2::Real)
        vcat(value(x1), value(x2)), (Δ) -> (Δ[1:length(x1)], Δ[length(x1)+1])
    end

    logabsdetjac(b::Log{1}, x::Union{RTV, RTM}) = track(logabsdetjac, b, x)
    DistributionsAD.ReverseDiffX.@grad function logabsdetjac(b::Log{1}, x::AbstractVector)
        return -sum(log, value(x)), Δ -> (nothing, -Δ ./ value(x))
    end
    DistributionsAD.ReverseDiffX.@grad function logabsdetjac(b::Log{1}, x::AbstractMatrix)
        return -vec(sum(log, value(x); dims = 1)), Δ -> (nothing, .- Δ' ./ value(x))
    end

    function _logabsdetjac_scale(a::RTR, x::Real, ::Val{0})
        return track(_logabsdetjac_scale, a, value(x), Val(0))
    end
    DistributionsAD.ReverseDiffX.@grad function _logabsdetjac_scale(a::Real, x::Real, v::Val{0})
        return _logabsdetjac_scale(value(a), value(x), Val(0)), Δ -> (inv(value(a)) .* Δ, nothing, nothing)
    end
    # Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
    function _logabsdetjac_scale(a::RTR, x::AbstractVector, ::Val{0})
        return track(_logabsdetjac_scale, a, value(x), Val(0))
    end
    DistributionsAD.ReverseDiffX.@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, v::Val{0})
        da = value(a)
        J = fill(inv.(da), length(x))
        return _logabsdetjac_scale(da, value(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
    end
    function _logabsdetjac_scale(a::RTR, x::AbstractMatrix, ::Val{0})
        return track(_logabsdetjac_scale, a, value(x), Val(0))
    end
    DistributionsAD.ReverseDiffX.@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, v::Val{0})
        da = value(a)
        J = fill(size(x, 1) / da, size(x, 2))
        return _logabsdetjac_scale(da, value(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
    end
    # adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
    function _logabsdetjac_scale(a::RTV, x::AbstractVector, ::Val{1})
        return track(_logabsdetjac_scale, a, value(x), Val(1))
    end
    DistributionsAD.ReverseDiffX.@grad function _logabsdetjac_scale(a::RTV, x::AbstractVector, v::Val{1})
        # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
        #                 = ∂ᵢ log |aᵢ|
        #                 = (1 / aᵢ) ∂ᵢ aᵢ
        #                 = (1 / aᵢ)
        da = value(a)
        J = inv.(da)
        return _logabsdetjac_scale(da, value(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
    end
    function _logabsdetjac_scale(a::RTV, x::AbstractMatrix, ::Val{1})
        return track(_logabsdetjac_scale, a, value(x), Val(1))
    end
    DistributionsAD.ReverseDiffX.@grad function _logabsdetjac_scale(a::RTV, x::AbstractMatrix, v::Val{1})
        da = value(a)
        Jᵀ = repeat(inv.(da), 1, size(x, 2))
        return _logabsdetjac_scale(da, value(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
    end

    function _simplex_bijector(X::Union{RTV, RTM}, b::SimplexBijector)
        return track(_simplex_bijector, X, b)
    end
    DistributionsAD.ReverseDiffX.@grad function _simplex_bijector(Y::AbstractVector, b::SimplexBijector)
        Yd = value(Y)
        return _simplex_bijector(Yd, b), Δ -> (simplex_link_jacobian(Yd)' * Δ, nothing)
    end
    DistributionsAD.ReverseDiffX.@grad function _simplex_bijector(Y::AbstractMatrix, b::SimplexBijector)
        Yd = value(Y)
        return _simplex_bijector(Yd, b), Δ -> begin
            maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
                simplex_link_jacobian(c1)' * c2
            end, nothing
        end
    end

    function _simplex_inv_bijector(X::Union{RTV, RTM}, b::SimplexBijector)
        return track(_simplex_inv_bijector, X, b)
    end
    DistributionsAD.ReverseDiffX.@grad function _simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector)
        Yd = value(Y)
        return _simplex_inv_bijector(Yd, b), Δ -> (simplex_invlink_jacobian(Yd)' * Δ, nothing)
    end
    DistributionsAD.ReverseDiffX.@grad function _simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector)
        Yd = value(Y)
        return _simplex_inv_bijector(Yd, b), Δ -> begin
            maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
                simplex_invlink_jacobian(c1)' * c2
            end, nothing
        end
    end

    replace_diag(::typeof(log), X::RTM) = track(replace_diag, log, X)
    DistributionsAD.ReverseDiffX.@grad function replace_diag(::typeof(log), X)
        Xd = value(X)
        f(i, j) = i == j ? log(Xd[i, j]) : Xd[i, j]
        out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
        out, ∇ -> begin
            g(i, j) = i == j ? ∇[i, j]/Xd[i, j] : ∇[i, j]
            return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
        end
    end

    replace_diag(::typeof(exp), X::RTM) = track(replace_diag, exp, X)
    DistributionsAD.ReverseDiffX.@grad function replace_diag(::typeof(exp), X)
        Xd = value(X)
        f(i, j) = ifelse(i == j, exp(Xd[i, j]), Xd[i, j])
        out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
        out, ∇ -> begin
            g(i, j) = ifelse(i == j, ∇[i, j]*exp(Xd[i, j]), ∇[i, j])
            return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
        end
    end

    logabsdetjac(b::SimplexBijector, x::Union{RTV, RTM}) = track(logabsdetjac, b, x)
    DistributionsAD.ReverseDiffX.@grad function logabsdetjac(b::SimplexBijector, x::AbstractVector)
        xd = value(x)
        return logabsdetjac(b, xd), Δ -> begin
            (nothing, simplex_logabsdetjac_gradient(xd) * Δ)
        end
    end
    DistributionsAD.ReverseDiffX.@grad function logabsdetjac(b::SimplexBijector, x::AbstractMatrix)
        xd = value(x)
        return logabsdetjac(b, xd), Δ -> begin
            (nothing, maphcat(eachcol(xd), Δ) do c, g
                simplex_logabsdetjac_gradient(c) * g
            end)
        end
    end
end

end

using .ReverseDiffCompat