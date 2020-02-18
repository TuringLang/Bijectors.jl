import .Tracker
using .Tracker: TrackedReal, TrackedVector, TrackedMatrix, TrackedArray, @grad

_eps(::Type{<:TrackedReal{T}}) where {T} = eps(T)

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inversed{<:ADBijector{<:TrackerAD}}},
    x::Real
)
    return Tracker.data(Tracker.gradient(b, x)[1])
end
function jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inversed{<:ADBijector{<:TrackerAD}}},
    x::AbstractVector{<:Real}
)
    # We extract `data` so that we don't return a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, x))
end

# implementations for Shift bijector
function _logabsdetjac_shift(a::TrackedReal, x::Real, ::Val{0})
    return Tracker.param(_logabsdetjac_shift(Tracker.data(a), Tracker.data(x), Val(0)))
end
function _logabsdetjac_shift(a::TrackedReal, x::AbstractVector{<:Real}, ::Val{0})
    return Tracker.param(_logabsdetjac_shift(Tracker.data(a), Tracker.data(x), Val(0)))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractVector{<:Real},
    ::Val{1}
)
    return Tracker.param(_logabsdetjac_shift(Tracker.data(a), Tracker.data(x), Val(1)))
end
function _logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractMatrix{<:Real},
    ::Val{1}
)
    return Tracker.param(_logabsdetjac_shift(Tracker.data(a), Tracker.data(x), Val(1)))
end

# implementations for Scale bijector
# Adjoints for 0-dim and 1-dim `Scale` using `Real`
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::Real, ::Val{0})
    return _logabsdetjac_scale(Tracker.data(a), Tracker.data(x), Val(0)), Δ -> (inv(Tracker.data(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0})
    da = Tracker.data(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, Tracker.data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{0})
    da = Tracker.data(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, Tracker.data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    da = Tracker.data(a)
    J = inv.(da)
    return _logabsdetjac_scale(da, Tracker.data(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    return Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    da = Tracker.data(a)
    Jᵀ = repeat(inv.(da), 1, size(x, 2))
    return _logabsdetjac_scale(da, Tracker.data(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end
# TODO: implement analytical gradient for scaling a vector using a matrix
# function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     Tracker.track(_logabsdetjac_scale, a, Tracker.data(x), Val{1})
# end
# @grad function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     throw
# end

# implementations for Stacked bijector
function logabsdetjac(b::Stacked, x::TrackedMatrix{<:Real})
    return Tracker.collect([logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)])
end
# TODO: implement custom adjoint since we can exploit block-diagonal nature of `Stacked`
function (sb::Stacked)(x::TrackedMatrix{<:Real})
    init = reshape(sb(x[:, 1]), :, 1)
    return Tracker.collect(mapreduce(i -> sb(x[:, i]), hcat, 2:size(x, 2); init = init))
end

# Stuff
function _simplex_bijector(b::SimplexBijector{proj}, x::TrackedVector{T}) where {proj, T}
    Tracker.track(_simplex_bijector, b, x)
end

Tracker.@grad function _simplex_bijector(b::SimplexBijector{proj}, x::AbstractVector{T}) where {proj, T}
    x_untracked = Tracker.data(x)
    return _simplex_bijector(b, x_untracked), Δ -> (nothing, jacobian(b, x_untracked)' * Δ)
end

function _simplex_bijector_inv(ib::Inversed{<:SimplexBijector{proj}}, y::TrackedVector{T}) where {proj, T}
    Tracker.track(_simplex_bijector_inv, ib, y)
end

Tracker.@grad function _simplex_bijector_inv(ib::Inversed{<:SimplexBijector{proj}}, y::AbstractVector{T}) where {proj, T}
    y_untracked = Tracker.data(y)
    return _simplex_bijector_inv(ib, y_untracked), Δ -> (nothing, jacobian(ib, y_untracked)' * Δ)
end
