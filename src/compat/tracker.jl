module TrackerCompat

using ..Tracker: Tracker,
                 TrackedReal,
                 TrackedVector,
                 TrackedMatrix,
                 TrackedArray,
                 TrackedVecOrMat,
                 @grad,
                 track,
                 data,
                 param

import ..Bijectors
using ..Bijectors: Log, SimplexBijector, ADBijector,
    TrackerAD, Inverse, Stacked, Exp

import ChainRulesCore
import LogExpFunctions

using Compat: eachcol
using LinearAlgebra
using Distributions: LocationScale

Bijectors.maporbroadcast(f, x::TrackedArray...) = f.(x...)
function Bijectors.maporbroadcast(
    f,
    x1::TrackedArray{T, N},
    x::AbstractArray{<:TrackedReal}...,
) where {T, N}
    return f.(convert(Array{TrackedReal{T}, N}, x1), x...)
end

Bijectors._eps(::Type{<:TrackedReal{T}}) where {T} = Bijectors._eps(T)
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

# AD implementations
function Bijectors.jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inverse{<:ADBijector{<:TrackerAD}}},
    x::Real
)
    return data(Tracker.gradient(b, x)[1])
end
function Bijectors.jacobian(
    b::Union{<:ADBijector{<:TrackerAD}, Inverse{<:ADBijector{<:TrackerAD}}},
    x::AbstractVector{<:Real}
)
    # We extract `data` so that we don't return a `Tracked` type
    return data(Tracker.jacobian(b, x))
end

# implementations for Shift bijector
function Bijectors._logabsdetjac_shift(a::TrackedReal, x::Real, ::Val{0})
    return tracker_shift_logabsdetjac(a, x, Val(0))
end
function Bijectors._logabsdetjac_shift(a::TrackedReal, x::AbstractVector{<:Real}, ::Val{0})
    return tracker_shift_logabsdetjac(a, x, Val(0))
end
function Bijectors._logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractVector{<:Real},
    ::Val{1}
)
    return tracker_shift_logabsdetjac(a, x, Val(1))
end
function Bijectors._logabsdetjac_shift(
    a::Union{TrackedReal, TrackedVector{<:Real}},
    x::AbstractMatrix{<:Real},
    ::Val{1}
)
    return tracker_shift_logabsdetjac(a, x, Val(1))
end
function tracker_shift_logabsdetjac(a, x, ::Val{N}) where {N}
    return param(Bijectors._logabsdetjac_shift(data(a), data(x), Val(N)))
end

# Log bijector

@grad function Bijectors.logabsdetjac(b::Log{1}, x::AbstractVector)
    return -sum(log, data(x)), Δ -> (nothing, -Δ ./ data(x))
end
@grad function Bijectors.logabsdetjac(b::Log{1}, x::AbstractMatrix)
    return -vec(sum(log, data(x); dims = 1)), Δ -> (nothing, .- Δ' ./ data(x))
end
@grad function Bijectors.logabsdetjac(b::Log{2}, x::AbstractMatrix)
    return -sum(log, data(x)), Δ -> (nothing, -Δ ./ data(x))
end

# implementations for Scale bijector
# Adjoints for 0-dim and 1-dim `Scale` using `Real`
function Bijectors._logabsdetjac_scale(a::TrackedReal, x::Real, ::Val{0})
    return track(Bijectors._logabsdetjac_scale, a, data(x), Val(0))
end
@grad function Bijectors._logabsdetjac_scale(a::Real, x::Real, ::Val{0})
    return Bijectors._logabsdetjac_scale(data(a), data(x), Val(0)), Δ -> (inv(data(a)) .* Δ, nothing, nothing)
end
# Need to treat `AbstractVector` and `AbstractMatrix` separately due to ambiguity errors
function Bijectors._logabsdetjac_scale(a::TrackedReal, x::AbstractVector, ::Val{0})
    return track(Bijectors._logabsdetjac_scale, a, data(x), Val(0))
end
@grad function Bijectors._logabsdetjac_scale(a::Real, x::AbstractVector, ::Val{0})
    da = data(a)
    J = fill(inv.(da), length(x))
    return Bijectors._logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
function Bijectors._logabsdetjac_scale(a::TrackedReal, x::AbstractMatrix, ::Val{0})
    return track(Bijectors._logabsdetjac_scale, a, data(x), Val(0))
end
@grad function Bijectors._logabsdetjac_scale(a::Real, x::AbstractMatrix, ::Val{0})
    da = data(a)
    J = fill(size(x, 1) / da, size(x, 2))
    return Bijectors._logabsdetjac_scale(da, data(x), Val(0)), Δ -> (transpose(J) * Δ, nothing, nothing)
end
# adjoints for 1-dim and 2-dim `Scale` using `AbstractVector`
function Bijectors._logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    return track(Bijectors._logabsdetjac_scale, a, data(x), Val(1))
end
@grad function Bijectors._logabsdetjac_scale(a::TrackedVector, x::AbstractVector, ::Val{1})
    # ∂ᵢ (∑ⱼ log|aⱼ|) = ∑ⱼ δᵢⱼ ∂ᵢ log|aⱼ|
    #                 = ∂ᵢ log |aᵢ|
    #                 = (1 / aᵢ) ∂ᵢ aᵢ
    #                 = (1 / aᵢ)
    da = data(a)
    J = inv.(da)
    return Bijectors._logabsdetjac_scale(da, data(x), Val(1)), Δ -> (J .* Δ, nothing, nothing)
end
function Bijectors._logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    return track(Bijectors._logabsdetjac_scale, a, data(x), Val(1))
end
@grad function Bijectors._logabsdetjac_scale(a::TrackedVector, x::AbstractMatrix, ::Val{1})
    da = data(a)
    Jᵀ = repeat(inv.(da), 1, size(x, 2))
    return Bijectors._logabsdetjac_scale(da, data(x), Val(1)), Δ -> (Jᵀ * Δ, nothing, nothing)
end
# TODO: implement analytical gradient for scaling a vector using a matrix
# function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     track(_logabsdetjac_scale, a, data(x), Val{1})
# end
# @grad function _logabsdetjac_scale(a::TrackedMatrix, x::AbstractVector, ::Val{1})
#     throw
# end
# implementations for Stacked bijector
function Bijectors.logabsdetjac(b::Stacked, x::TrackedMatrix{<:Real})
    return map(eachcol(x)) do c
        Bijectors.logabsdetjac(b, c)
    end
end
# TODO: implement custom adjoint since we can exploit block-diagonal nature of `Stacked`
function (sb::Stacked)(x::TrackedMatrix{<:Real})
    return Bijectors.eachcolmaphcat(sb, x)
end
# Simplex adjoints
function Bijectors._simplex_bijector(X::TrackedVecOrMat, b::SimplexBijector{1})
    return track(Bijectors._simplex_bijector, X, b)
end
function Bijectors._simplex_inv_bijector(Y::TrackedVecOrMat, b::SimplexBijector{1})
    return track(Bijectors._simplex_inv_bijector, Y, b)
end
@grad function Bijectors._simplex_bijector(X::AbstractVector, b::SimplexBijector{1})
    Xd = data(X)
    return Bijectors._simplex_bijector(Xd, b), Δ -> (Bijectors.simplex_link_jacobian(Xd)' * Δ, nothing)
end
@grad function Bijectors._simplex_inv_bijector(Y::AbstractVector, b::SimplexBijector{1})
    Yd = data(Y)
    return Bijectors._simplex_inv_bijector(Yd, b), Δ -> (Bijectors.simplex_invlink_jacobian(Yd)' * Δ, nothing)
end

@grad function Bijectors._simplex_bijector(X::AbstractMatrix, b::SimplexBijector{1})
    Xd = data(X)
    return Bijectors._simplex_bijector(Xd, b), Δ -> begin
        Bijectors.maphcat(eachcol(Xd), eachcol(Δ)) do c1, c2
            Bijectors.simplex_link_jacobian(c1)' * c2
        end, nothing
    end
end
@grad function Bijectors._simplex_inv_bijector(Y::AbstractMatrix, b::SimplexBijector{1})
    Yd = data(Y)
    return Bijectors._simplex_inv_bijector(Yd, b), Δ -> begin
        Bijectors.maphcat(eachcol(Yd), eachcol(Δ)) do c1, c2
            Bijectors.simplex_invlink_jacobian(c1)' * c2
        end, nothing
    end
end

Bijectors.replace_diag(::typeof(log), X::TrackedMatrix) = track(Bijectors.replace_diag, log, X)
@grad function Bijectors.replace_diag(::typeof(log), X)
    Xd = data(X)
    f(i, j) = i == j ? log(Xd[i, j]) : Xd[i, j]
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = i == j ? ∇[i, j]/Xd[i, j] : ∇[i, j]
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

Bijectors.replace_diag(::typeof(exp), X::TrackedMatrix) = track(Bijectors.replace_diag, exp, X)
@grad function Bijectors.replace_diag(::typeof(exp), X)
    Xd = data(X)
    f(i, j) = ifelse(i == j, exp(Xd[i, j]), Xd[i, j])
    out = f.(1:size(Xd, 1), (1:size(Xd, 2))')
    out, ∇ -> begin
        g(i, j) = ifelse(i == j, ∇[i, j]*exp(Xd[i, j]), ∇[i, j])
        return (nothing, g.(1:size(Xd, 1), (1:size(Xd, 2))'))
    end
end

Bijectors.logabsdetjac(b::SimplexBijector{1}, x::TrackedVecOrMat) = track(Bijectors.logabsdetjac, b, x)
@grad function Bijectors.logabsdetjac(b::SimplexBijector{1}, x::AbstractVector)
    xd = data(x)
    return Bijectors.logabsdetjac(b, xd), Δ -> begin
        (nothing, Bijectors.simplex_logabsdetjac_gradient(xd) * Δ)
    end
end
@grad function Bijectors.logabsdetjac(b::SimplexBijector{1}, x::AbstractMatrix)
    xd = data(x)
    return Bijectors.logabsdetjac(b, xd), Δ -> begin
        (nothing, Bijectors.maphcat(eachcol(xd), Δ) do c, g
            Bijectors.simplex_logabsdetjac_gradient(c) * g
        end)
    end
end

for header in [
    (:(α_::TrackedReal), :β, :z_0, :(z::AbstractVector)),
    (:α_, :(β::TrackedReal), :z_0, :(z::AbstractVector)),
    (:α_, :β, :(z_0::TrackedVector), :(z::AbstractVector)),
    (:α_, :β, :z_0, :(z::TrackedVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::AbstractVector)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::AbstractVector)),
    (:(α_::TrackedReal), :β, :z_0, :(z::TrackedVecOrMat)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::AbstractVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::TrackedVector)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::TrackedVector)),
    (:α_, :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedVector)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedVector)),
]
    @eval begin
        function Bijectors._radial_transform($(header...))
            α = LogExpFunctions.log1pexp(α_)            # from A.2
            β_hat = -α + LogExpFunctions.log1pexp(β)    # from A.2
            if β_hat isa TrackedReal
                TV = vectorof(typeof(β_hat))
                T = vectorof(typeof(β_hat))
            elseif z_0 isa TrackedVector
                TV = typeof(z_0)
                T = typeof(z_0)
            else
                T = TV = typeof(z)
            end
            Tr = promote_type(eltype(z), eltype(z_0))
            r::Tr = norm((z .- z_0)::TV)
            transformed::T = z .+ β_hat ./ (α .+ r') .* (z .- z_0)   # from eq(14)
            return (transformed = transformed, α = α, β_hat = β_hat, r = r)
        end
    end
end

for header in [
    (:(α_::TrackedReal), :β, :z_0, :(z::AbstractMatrix)),
    (:α_, :(β::TrackedReal), :z_0, :(z::AbstractMatrix)),
    (:α_, :β, :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:α_, :β, :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :β, :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::AbstractMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :z_0, :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :β, :(z_0::TrackedVector), :(z::TrackedMatrix)),
    (:α_, :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedMatrix)),
    (:(α_::TrackedReal), :(β::TrackedReal), :(z_0::TrackedVector), :(z::TrackedMatrix)),
]
    @eval begin
        function Bijectors._radial_transform($(header...))
            α = LogExpFunctions.log1pexp(α_)            # from A.2
            β_hat = -α + LogExpFunctions.log1pexp(β)    # from A.2
            if β_hat isa TrackedReal
                TV = vectorof(typeof(β_hat))
                T = matrixof(TV)
            elseif z_0 isa TrackedVector
                TV = typeof(z_0)
                T = matrixof(TV)
            else
                T = typeof(z)
                TV = vectorof(T)
            end
            r::TV = eachcolnorm(z .- z_0)
            transformed::T = z .+ β_hat ./ (α .+ r') .* (z .- z_0)   # from eq(14)
            return (transformed = transformed, α = α, β_hat = β_hat, r = r)
        end
    end
end
eachcolnorm(X) = map(norm, eachcol(X))
eachcolnorm(X::TrackedMatrix) = track(eachcolnorm, X)
@grad function eachcolnorm(X)
    Xd = data(X)
    y = map(norm, eachcol(Xd))
    y, Δ -> begin
        (Xd .* (Δ ./ y)',)
    end
end

function matrixof(::Type{<:TrackedArray{T,1,Vector{T}}}) where {T<:Real}
    return TrackedArray{T,2,Matrix{T}}
end
function matrixof(::Type{TrackedReal{T}}) where {T<:Real}
    return TrackedArray{T,2,Matrix{T}}
end
function vectorof(::Type{<:TrackedArray{T,2,Matrix{T}}}) where {T<:Real}
    return TrackedArray{T,1,Vector{T}}
end
function vectorof(::Type{TrackedReal{T}}) where {T<:Real}
    return TrackedArray{T,1,Vector{T}}
end

(b::Exp{0})(x::TrackedVector) = exp.(x)::vectorof(float(eltype(x)))
(b::Exp{1})(x::TrackedVector) = exp.(x)::vectorof(float(eltype(x)))
(b::Exp{1})(x::TrackedMatrix) = exp.(x)::matrixof(float(eltype(x)))
(b::Exp{2})(x::TrackedMatrix) = exp.(x)::matrixof(float(eltype(x)))

(b::Log{0})(x::TrackedVector) = log.(x)::vectorof(float(eltype(x)))
(b::Log{1})(x::TrackedVector) = log.(x)::vectorof(float(eltype(x)))
(b::Log{1})(x::TrackedMatrix) = log.(x)::matrixof(float(eltype(x)))
(b::Log{2})(x::TrackedMatrix) = log.(x)::matrixof(float(eltype(x)))

Bijectors.logabsdetjac(b::Log{0}, x::TrackedVector) = .-log.(x)::vectorof(float(eltype(x)))
Bijectors.logabsdetjac(b::Log{1}, x::TrackedMatrix) = - vec(sum(log.(x); dims = 1))

Bijectors.getpd(X::TrackedMatrix) = track(Bijectors.getpd, X)
@grad function Bijectors.getpd(X::AbstractMatrix)
    Xd = data(X)
    return Bijectors.LowerTriangular(Xd) * Bijectors.LowerTriangular(Xd)', Δ -> begin
        Xl = Bijectors.LowerTriangular(Xd)
        return (Bijectors.LowerTriangular(Δ' * Xl + Δ * Xl),)
    end
end

Bijectors.lower(A::TrackedMatrix) = track(Bijectors.lower, A)
@grad function Bijectors.lower(A::AbstractMatrix)
    Ad = data(A)
    return Bijectors.lower(Ad), Δ -> (Bijectors.lower(Δ),)
end

Bijectors._inv_link_chol_lkj(y::TrackedMatrix) = track(Bijectors._inv_link_chol_lkj, y)
@grad function Bijectors._inv_link_chol_lkj(y_tracked)
    y = data(y_tracked)

    K = LinearAlgebra.checksquare(y)

    w = similar(y)

    z_mat = similar(y) # cache for adjoint
    tmp_mat = similar(y)
    
    @inbounds for j in 1:K
        w[1, j] = 1
        for i in 2:j
            z = tanh(y[i-1, j])
            tmp = w[i-1, j]

            z_mat[i, j] = z
            tmp_mat[i, j] = tmp

            w[i-1, j] = z * tmp
            w[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j+1):K
            w[i, j] = 0
        end
    end

    function pullback_inv_link_chol_lkj(Δw)
        LinearAlgebra.checksquare(Δw)

        Δy = zero(y)

        @inbounds for j in 1:K
            Δtmp = Δw[j,j]
            for i in j:-1:2
                Δz = Δw[i-1, j] * tmp_mat[i, j] - Δtmp * tmp_mat[i, j] / sqrt(1 - z_mat[i, j]^2) * z_mat[i, j]
                Δy[i-1, j] = Δz / cosh(y[i-1, j])^2
                Δtmp = Δw[i-1, j] * z_mat[i, j] + Δtmp * sqrt(1 - z_mat[i, j]^2)
            end
        end
        
        return (Δy,)
    end

    return w, pullback_inv_link_chol_lkj
end

Bijectors._link_chol_lkj(w::TrackedMatrix) = track(Bijectors._link_chol_lkj, w)
@grad function Bijectors._link_chol_lkj(w_tracked)
    w = data(w_tracked)

    K = LinearAlgebra.checksquare(w)
    
    z = similar(w)

    @inbounds z[1, 1] = 0

    tmp_mat = similar(w) # cache for pullback.

    @inbounds for j=2:K
        z[1, j] = atanh(w[1, j])
        tmp = sqrt(1 - w[1, j]^2)
        tmp_mat[1, j] = tmp
        for i in 2:(j - 1)
            p = w[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            tmp_mat[i, j] = tmp
            z[i, j] = atanh(p)
        end
        z[j, j] = 0
    end

    function pullback_link_chol_lkj(Δz)
        LinearAlgebra.checksquare(Δz)

        Δw = similar(w)

        @inbounds Δw[1,1] = zero(eltype(Δz))

        @inbounds for j=2:K
            Δw[j, j] = 0
            Δtmp = zero(eltype(Δz)) # Δtmp_mat[j-1,j]
            for i in (j-1):-1:2
                p = w[i, j] / tmp_mat[i-1, j]
                ftmp = sqrt(1 - p^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -w[i,j] / tmp_mat[i-1, j]^2

                Δp = Δz[i,j] / (1-p^2) + Δtmp * tmp_mat[i-1, j] * d_ftmp_p
                Δw[i, j] = Δp / tmp_mat[i-1, j]
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp # update to "previous" Δtmp
            end
            Δw[1, j] = Δz[1, j] / (1-w[1,j]^2) - Δtmp / sqrt(1 - w[1,j]^2) * w[1,j]
        end

        return (Δw,)
    end

    return z, pullback_link_chol_lkj
end

function Bijectors.find_alpha(wt_y::T, wt_u_hat::T, b::T) where {T<:TrackedReal}
    return track(Bijectors.find_alpha, wt_y, wt_u_hat, b)
end
@grad function Bijectors.find_alpha(wt_y::TrackedReal, wt_u_hat::TrackedReal, b::TrackedReal)
    α = Bijectors.find_alpha(data(wt_y), data(wt_u_hat), data(b))

    ∂wt_y = inv(1 + wt_u_hat * sech(α + b)^2)
    ∂wt_u_hat = - tanh(α + b) * ∂wt_y
    ∂b = ∂wt_y - 1
    find_alpha_pullback(Δ::Real) = (Δ * ∂wt_y, Δ * ∂wt_u_hat, Δ * ∂b)

    return α, find_alpha_pullback
end

# `OrderedBijector`
Bijectors._transform_ordered(y::Union{TrackedVector,TrackedMatrix}) = track(Bijectors._transform_ordered, y)
@grad function Bijectors._transform_ordered(y::AbstractVecOrMat)
    x, dx = ChainRulesCore.rrule(Bijectors._transform_ordered, data(y))
    return x, (wrap_chainrules_output ∘ Base.tail ∘ dx)
end

function Bijectors._transform_inverse_ordered(x::Union{TrackedVector,TrackedMatrix})
    return track(Bijectors._transform_inverse_ordered, x)
end
@grad function Bijectors._transform_inverse_ordered(x::AbstractVecOrMat)
    y, dy = ChainRulesCore.rrule(Bijectors._transform_inverse_ordered, data(x))
    return y, (wrap_chainrules_output ∘ Base.tail ∘ dy)
end

# NOTE: Probably doesn't work in complete generality.
wrap_chainrules_output(x) = x
wrap_chainrules_output(x::ChainRulesCore.AbstractZero) = nothing
wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)

end
