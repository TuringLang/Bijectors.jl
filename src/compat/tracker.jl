module TrackerCompat

using ..Bijectors: ADBijector, TrackerAD, Inversed, Stacked
import ..Bijectors: _eps, _jacobian, _logabsdetjac_shift, _logabsdetjac_scale

using ..Tracker: TrackedReal, TrackedVector, TrackedMatrix, TrackedArray,
        track, data, gradient, param, collectmemaybe
# we do not use the `@grad` macro but instead implement `_forward`
# since `@grad` calls `Tracker._forward` whereas we need `..Tracker._forward`
import ..Tracker: _forward

_eps(::Type{<:TrackedReal{T}}) where {T} = eps(T)

# AD implementations
function _jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inversed{<:ADBijector{<:TrackerAD}}},
    x::Real
)
    return data(gradient(b, x)[1])
end
function _jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inversed{<:ADBijector{<:TrackerAD}}},
    x::AbstractVector{<:Real}
)
    # We extract `data` so that we don't return a `Tracked` type
    return data(jacobian(b, x))
end

# implementations for Shift bijector
function _logabsdetjac_shift(a::TrackedReal, x::Real, ::Val{0})
    return param(_logabsdetjac_shift(data(a), data(x), Val(0)))
end
function _logabsdetjac_shift(a::TrackedReal, x::AbstractVector{<:Real}, ::Val{0})
    return param(_logabsdetjac_shift(data(a), data(x), Val(0)))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractVector{<:Real},
    ::Val{1}
)
    return param(_logabsdetjac_shift(data(a), data(x), Val(1)))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractMatrix{<:Real},
    ::Val{1}
)
    return param(_logabsdetjac_shift(data(a), data(x), Val(1)))
end

# implementations for Scale bijector
# Adjoints for 0-dim and 1-dim `Scale` using `Real`
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
function _forward(::typeof(_logabsdetjac_scale), a::Real, x::Real, ::Val{0})
    return _logabsdetjac_scale(data(a), data(x), Val(0)), Δ -> (inv(data(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
function _forward(::typeof(_logabsdetjac_scale), a::Real, x::AbstractVector, ::Val{0})
    da = data(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
function _forward(::typeof(_logabsdetjac_scale), a::Real, x::AbstractMatrix, ::Val{0})
    da = data(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return track(_logabsdetjac_scale, a, data(x), Val(1))
end
function _forward(::typeof(_logabsdetjac_scale), a::TrackedVector, x::AbstractVector, ::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    da = data(a)
    J = inv.(da)
    return _logabsdetjac_scale(da, data(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    return track(_logabsdetjac_scale, a, data(x), Val(1))
end
function _forward(::typeof(_logabsdetjac_scale), a::TrackedVector, x::AbstractMatrix, ::Val{1})
    da = data(a)
    Jᵀ = repeat(inv.(da), 1, size(x, 2))
    return _logabsdetjac_scale(da, data(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end
# TODO: implement analytical gradient for scaling a vector using a matrix
# function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     track(_logabsdetjac_scale, a, data(x), Val{1})
# end
# @grad function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     throw
# end

# implementations for Stacked bijector
function logabsdetjac(b::Stacked, x::TrackedMatrix{<:Real})
    return collectmemaybe([logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)])
end
# TODO: implement custom adjoint since we can exploit block-diagonal nature of `Stacked`
function (sb::Stacked)(x::TrackedMatrix{<:Real})
    init = reshape(sb(x[:, 1]), :, 1)
    return collectmemaybe(mapreduce(i -> sb(x[:, i]), hcat, 2:size(x, 2); init = init))
end

end