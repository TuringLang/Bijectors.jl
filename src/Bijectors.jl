module Bijectors

using Reexport, Requires
@reexport using Distributions
using StatsFuns
using LinearAlgebra
using MappedArrays
using Roots

export  TransformDistribution,
        PositiveDistribution,
        UnitDistribution,
        SimplexDistribution,
        PDMatDistribution,
        link,
        invlink,
        logpdf_with_trans,
        transform,
        forward,
        logabsdetjac,
        logabsdetjacinv,
        Bijector,
        ADBijector,
        Inversed,
        Composed,
        compose,
        Stacked,
        Identity,
        DistributionBijector,
        bijector,
        transformed,
        TransformedDistribution,
        UnivariateTransformed,
        MultivariateTransformed,
        entropy,
        logpdf_with_jac,
        logpdf_forward,
        PlanarLayer,
        RadialLayer,

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_BIJECTORS", "0")))

# Workaround for eps(::ForwardDiff.Dual)
_eps(::Type{T}) where {T} = T(eps(T))
_eps(::Type{Real}) = eps(Float64)
function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" @eval begin
        _eps(::Type{<:ForwardDiff.Dual{<:Any, Real}}) = _eps(Real)
    end
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" @eval begin
        _eps(::Type{<:Flux.Tracker.TrackedReal{T}}) where {T} = eps(T)
    end
end

#=
  NOTE: Codes below are adapted from
  https://github.com/brian-j-smith/Mamba.jl/blob/master/src/distributions/transformdistribution.jl
  The Mamba.jl package is licensed under the MIT License:
  > Copyright (c) 2014: Brian J Smith and other contributors:
  >
  > https://github.com/brian-j-smith/Mamba.jl/contributors
  >
  > Permission is hereby granted, free of charge, to any person obtaining
  > a copy of this software and associated documentation files (the
  > "Software"), to deal in the Software without restriction, including
  > without limitation the rights to use, copy, modify, merge, publish,
  > distribute, sublicense, and/or sell copies of the Software, and to
  > permit persons to whom the Software is furnished to do so, subject to
  > the following conditions:
  >
  > The above copyright notice and this permission notice shall be
  > included in all copies or substantial portions of the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#############
# a ≦ x ≦ b #
#############

const TransformDistribution{T<:ContinuousUnivariateDistribution} = Union{T, Truncated{T}}
@inline function _clamp(x::T, dist::TransformDistribution) where {T <: Real}
    ϵ = _eps(T)
    bounds = (minimum(dist) + ϵ, maximum(dist) - ϵ)
    clamped_x = ifelse(x < bounds[1], bounds[1], ifelse(x > bounds[2], bounds[2], x))
    DEBUG && @debug "x = $x, bounds = $bounds, clamped_x = $clamped_x"
    return clamped_x
end

link(d::TransformDistribution, x::Real) = _link(d, _clamp(x, d))
function _link(d::TransformDistribution, x::Real)
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return StatsFuns.logit((x - a) / (b - a))
    elseif lowerbounded
        return log(x - a)
    elseif upperbounded
        return log(b - x)
    else
        return x
    end
end

invlink(d::TransformDistribution, y::Real) = _clamp(_invlink(d, y), d)
function _invlink(d::TransformDistribution, y::Real)
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return (b - a) * StatsFuns.logistic(y) + a
    elseif lowerbounded
        return exp(y) + a
    elseif upperbounded
        return b - exp(y)
    else
        return y
    end
end

function logpdf_with_trans(d::TransformDistribution, x::Real, transform::Bool)
    x = transform ? _clamp(x, d) : x
    return _logpdf_with_trans(d, x, transform)
end
function _logpdf_with_trans(d::TransformDistribution, x::Real, transform::Bool)
    lp = logpdf(d, x)
    if transform
        a, b = minimum(d), maximum(d)
        lowerbounded, upperbounded = isfinite(a), isfinite(b)
        if lowerbounded && upperbounded
            lp += log((x - a) * (b - x) / (b - a))
        elseif lowerbounded
            lp += log(x - a)
        elseif upperbounded
            lp += log(b - x)
        end
    end
    return lp
end

#########
# 0 < x #
#########

const PositiveDistribution = Union{
    BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma,
    InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull,
}

_link(d::PositiveDistribution, x::Real) = log(x)
_invlink(d::PositiveDistribution, y::Real) = exp(y)
function _logpdf_with_trans(d::PositiveDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x)
end


#############
# 0 < x < 1 #
#############

const UnitDistribution = Union{Beta, KSOneSided, NoncentralBeta}

_link(d::UnitDistribution, x::Real) = StatsFuns.logit(x)
_invlink(d::UnitDistribution, y::Real) = StatsFuns.logistic(y)
function _logpdf_with_trans(d::UnitDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x * (one(x) - x))
end


###########
# ∑xᵢ = 1 #
###########

const SimplexDistribution = Union{Dirichlet}
function _clamp(x::T, dist::SimplexDistribution) where T
    bounds = (zero(T), one(T))
    clamped_x = clamp(x, bounds...)
    DEBUG && @debug "x = $x, bounds = $bounds, clamped_x = $clamped_x"
    return clamped_x
end

function link(
    d::SimplexDistribution,
    x::AbstractVector{T},
    ::Type{Val{proj}} = Val{true}
) where {T<:Real, proj}
    y, K = similar(x), length(x)

    ϵ = _eps(T)
    sum_tmp = zero(T)
    @inbounds z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    @inbounds y[1] = StatsFuns.logit(z) + log(T(K - 1))
    @inbounds @simd for k in 2:(K - 1)
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

# Vectorised implementation of the above.
function link(
    d::SimplexDistribution,
    X::AbstractMatrix{T},
    ::Type{Val{proj}} = Val{true}
) where {T<:Real, proj}
    Y, K, N = similar(X), size(X, 1), size(X, 2)

    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
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

function invlink(
    d::SimplexDistribution,
    y::AbstractVector{T},
    ::Type{Val{proj}} = Val{true}
) where {T<:Real, proj}
    x, K = similar(y), length(y)

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), d)
    sum_tmp = zero(T)
    @inbounds @simd for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, d)
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        x[K] = _clamp(one(T) - sum_tmp, d)
    else
        x[K] = _clamp(one(T) - sum_tmp - y[K], d)
    end
    return x
end

# Vectorised implementation of the above.
function invlink(
    d::SimplexDistribution,
    Y::AbstractMatrix{T},
    ::Type{Val{proj}} = Val{true}
) where {T<:Real, proj}
    X, K, N = similar(Y), size(Y, 1), size(Y, 2)

    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp, z = zero(T), StatsFuns.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), d)
        for k in 2:(K - 1)
            z = StatsFuns.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, d)
        end
        sum_tmp += X[K - 1, n]
        if proj
            X[K, n] = _clamp(one(T) - sum_tmp, d)
        else
            X[K, n] = _clamp(one(T) - sum_tmp - Y[K, n], d)
        end
    end

    return X
end

function logpdf_with_trans(
    d::SimplexDistribution,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    T = eltype(x)
    ϵ = _eps(T)
    lp = logpdf(d, mappedarray(x->x+ϵ, x))
    if transform
        K = length(x)

        sum_tmp = zero(eltype(x))
        @inbounds z = x[1]
        lp += log(z + ϵ) + log((one(T) + ϵ) - z)
        @inbounds @simd for k in 2:(K - 1)
            sum_tmp += x[k-1]
            z = x[k] / ((one(T) + ϵ) - sum_tmp)
            lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
        end
    end
    return lp
end

# REVIEW: why do we put this piece of code here?
function logpdf_with_trans(d::Categorical, x::Int)
    return d.p[x] > 0.0 && insupport(d, x) ? log(d.p[x]) : eltype(d.p)(-Inf)
end


###############
# MvLogNormal #
###############

using Distributions: AbstractMvLogNormal

link(d::AbstractMvLogNormal, x::AbstractVector{<:Real}) = log.(x)
invlink(d::AbstractMvLogNormal, y::AbstractVector{<:Real}) = exp.(y)
function logpdf_with_trans(
    d::AbstractMvLogNormal,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    return logpdf(d, x) + transform * sum(log, x)
end

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{InverseWishart, Wishart}

function link(d::PDMatDistribution, X::AbstractMatrix{<:Real})
    Y = Matrix(cholesky(X).L)
    Y[diagind(Y)] .= log.(view(Y, diagind(Y)))
    return Y
end

function invlink(d::PDMatDistribution, Y::AbstractMatrix{<:Real})
    X = copy(Y)
    X[diagind(X)] .= exp.(view(X, diagind(X)))
    return LowerTriangular(X) * LowerTriangular(X)'
end

function logpdf_with_trans(
    d::PDMatDistribution,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    T = eltype(X)
    Xcf = cholesky(X, check=false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + (eps(T) * norm(X)) * I)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        U = Xcf.U
        @inbounds @simd for i in 1:dim(d)
            lp += (dim(d) - i + 2) * log(U[i, i])
        end
        lp += dim(d) * log(T(2))
    end
    return lp
end
function getlogp(d::Wishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.S \ X)) - d.c0
end
function getlogp(d::InverseWishart, Xcf, X)
    Ψ = Matrix(d.Ψ)
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end

############################################
# Defaults (assume identity link function) #
############################################

# UnivariateDistributions
using Distributions: UnivariateDistribution

link(d::UnivariateDistribution, x::Real) = x
link(d::UnivariateDistribution, x::AbstractVector{<:Real}) = map(x -> link(d, x), x)

invlink(d::UnivariateDistribution, y::Real) = y
invlink(d::UnivariateDistribution, y::AbstractVector{<:Real}) = map(y -> invlink(d, y), y)

logpdf_with_trans(d::UnivariateDistribution, x::Real, ::Bool) = logpdf(d, x)
function logpdf_with_trans(
    d::UnivariateDistribution,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    return map(x -> logpdf_with_trans(d, x, transform), x)
end

# MultivariateDistributions
using Distributions: MultivariateDistribution

link(d::MultivariateDistribution, x::AbstractVector{<:Real}) = copy(x)
link(d::MultivariateDistribution, X::AbstractMatrix{<:Real}) = copy(X)

invlink(d::MultivariateDistribution, y::AbstractVector{<:Real}) = copy(y)
invlink(d::MultivariateDistribution, Y::AbstractMatrix{<:Real}) = copy(Y)

function logpdf_with_trans(d::MultivariateDistribution, x::AbstractVector{<:Real}, ::Bool)
    return logpdf(d, x)
end
function logpdf_with_trans(
    d::MultivariateDistribution,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    return [logpdf_with_trans(d, view(X, :, n), transform) for n in 1:size(X, 2)]
end

# MatrixDistributions
using Distributions: MatrixDistribution

link(d::MatrixDistribution, X::AbstractMatrix{<:Real}) = copy(X)
link(d::MatrixDistribution, X::AbstractVector{<:AbstractMatrix{<:Real}}) = map(x -> link(d, x), X)

invlink(d::MatrixDistribution, Y::AbstractMatrix{<:Real}) = copy(Y)
function invlink(d::MatrixDistribution, Y::AbstractVector{<:AbstractMatrix{<:Real}})
    return map(y -> invlink(d, y), Y)
end

logpdf_with_trans(d::MatrixDistribution, X::AbstractMatrix{<:Real}, ::Bool) = logpdf(d, X)
function logpdf_with_trans(
    d::MatrixDistribution,
    X::AbstractVector{<:AbstractMatrix{<:Real}},
    transform::Bool,
)
    return map(x -> logpdf_with_trans(d, x, transform), X)
end

include("interface.jl")

include("norm_flows.jl")

end # module
