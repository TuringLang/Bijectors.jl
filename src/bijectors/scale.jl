struct Scale{T, N} <: Bijector{N}
    a::T
end

function Scale(a::Union{Real,AbstractArray}; dim::Val{D} = Val(ndims(a))) where D
    return Scale{typeof(a), D}(a)
end

(b::Scale)(x) = b.a .* x
(b::Scale{<:Real})(x::AbstractArray) = b.a .* x
(b::Scale{<:AbstractMatrix})(x::AbstractArray) = b.a * x
(b::Scale{<:AbstractVector{<:Real}, 2})(x::AbstractMatrix{<:Real}) = b.a .* x

inv(b::Scale{T, D}) where {T, D} = Scale(inv(b.a); dim = Val(D))
inv(b::Scale{<:AbstractVector, D}) where {D} = Scale(inv.(b.a); dim = Val(D))

# We're going to implement custom adjoint for this
logabsdetjac(b::Scale{T, N}, x) where {T, N} = _logabsdetjac_scale(b.a, x, Val(N))

_logabsdetjac_scale(a::Real, x::Real, ::Val{0}) = log(abs(a))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0}) = fill(log(abs(a)), length(x))
_logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{1}) = log(abs(a)) * length(x)
_logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{1}) = fill(log(abs(a)) * size(x, 1), size(x, 2))
_logabsdetjac_scale(a::AbstractVector, x::AbstractVector, ::Val{1}) = sum(log.(abs.(a)))
_logabsdetjac_scale(a::AbstractVector, x::AbstractMatrix, ::Val{1}) = fill(sum(log.(abs.(a))), size(x, 2))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractVector, ::Val{1}) = log(abs(det(a)))
_logabsdetjac_scale(a::AbstractMatrix, x::AbstractMatrix{T}, ::Val{1}) where {T} = log(abs(det(a))) * ones(T, size(x, 2))

# Adjoints for 0-dim and 1-dim `Scale` using `Real`
function _logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::Real, ::Val{0})
    return _logabsdetjac_scale(data(a), data(x), Val(0)), Δ -> (inv(data(a)) .* Δ, nothing, nothing)
end

# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function _logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0})
    da = data(a)
    J = fill(inv.(da), length(x))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end

function _logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(_logabsdetjac_scale, a, data(x), Val(0))
end
@grad function _logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{0})
    da = data(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return _logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end

# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return track(_logabsdetjac_scale, a, data(x), Val(1))
end
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
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
@grad function _logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
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
