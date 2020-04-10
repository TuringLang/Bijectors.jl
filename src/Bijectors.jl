module Bijectors

using Reexport, Requires
@reexport using Distributions
using StatsFuns
using LinearAlgebra
using MappedArrays
using Roots
using Base.Iterators: drop
using LinearAlgebra: AbstractTriangular

export  TransformDistribution,
        PositiveDistribution,
        UnitDistribution,
        SimplexDistribution,
        PDMatDistribution,
        link,
        invlink,
        logpdf_with_trans,
        isclosedform,
        transform,
        forward,
        logabsdetjac,
        logabsdetjacinv,
        Bijector,
        ADBijector,
        Inverse,
        Composed,
        compose,
        Stacked,
        stack,
        Identity,
        DistributionBijector,
        bijector,
        transformed,
        TransformedDistribution,
        UnivariateTransformed,
        MultivariateTransformed,
        logpdf_with_jac,
        logpdf_forward,
        PlanarLayer,
        RadialLayer,
        InvertibleBatchNorm

if VERSION < v"1.1"
    using Compat: eachcol
end

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_BIJECTORS", "0")))
_debug(str) = @debug str

_eps(::Type{T}) where {T} = T(eps(T))
_eps(::Type{Real}) = eps(Float64)
_eps(::Type{<:Integer}) = eps(Float64)

function mapvcat(f, args...)
    out = map(f, args...)
    init = vcat(out[1])
    return reshape(reduce(vcat, drop(out, 1); init = init), size(out))
end

function maphcat(f, args...)
    out = map(f, args...)
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init = init)
end
function eachcolmaphcat(f, x1, x2)
    out = [f(x1[:,i], x2[i]) for i in 1:size(x1, 2)]
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init = init)
end
function eachcolmaphcat(f, x)
    out = map(f, eachcol(x))
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init = init)
end
function sumeachcol(f, x1, x2)
    # Using a view below for x1 breaks Tracker
    return sum(f(x1[:,i], x2[i]) for i in 1:size(x1, 2))
end

# Fallback

function logpdf_with_trans(d::UnivariateDistribution, x::AbstractArray{<:Real}, trans::Bool)
    return _flat_logpdf_with_trans(d, x, trans)
end
function _flat_logpdf_with_trans(dist, x, trans)
    if toflatten(dist)
        f, args = flatten(dist, trans)
        return f.(args..., x)
    else
        return mapvcat(x) do x
            logpdf_with_trans(dist, x, trans)
        end
    end
end

# Discrete distributions

function logpdf_with_trans(d::DiscreteUnivariateDistribution, x::Integer, transform::Bool)
    return logpdf(d, x)
end

function logpdf_with_trans(
    d::DiscreteMultivariateDistribution,
    x::AbstractVecOrMat{<:Real},
    transform::Bool,
)
    return logpdf(d, x)
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
function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    ϵ = _eps(T)
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    DEBUG && _debug("x = $x, bounds = $((a, b)), clamped_x = $clamped_x")
    return clamped_x
end

function link(d::TransformDistribution, x::Real)
    a, b = minimum(d), maximum(d)
    return transformed_link(_clamp(x, a, b), a, b)
end
function link(d::TransformDistribution, x::AbstractArray{<:Real})
    a, b = minimum(d), maximum(d)
    return transformed_link.(_clamp.(x, a, b), a, b)
end
function transformed_link(x::Real, a, b)
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

function invlink(d::TransformDistribution, y::Real)
    a, b = minimum(d), maximum(d)
    _clamp(transformed_invlink(y, a, b), a, b)
end
function invlink(d::TransformDistribution, y::AbstractArray{<:Real})
    a, b = minimum(d), maximum(d)
    _clamp.(transformed_invlink.(y, a, b), a, b)
end
function transformed_invlink(y, a, b)
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

function logpdf_with_trans(
    d::TransformDistribution,
    x::Real,
    transform::Bool,
)
    lp = logpdf(d, x)
    a, b = minimum(d), maximum(d)
    if transform
        x = _clamp(x, a, b)
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

link(d::PositiveDistribution, x::Real) = log(x)
invlink(d::PositiveDistribution, y::Real) = exp(y)
link(d::PositiveDistribution, x::AbstractArray{<:Real}) = log.(x)
invlink(d::PositiveDistribution, y::AbstractArray{<:Real}) = exp.(y)
function logpdf_with_trans(d::PositiveDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x)
end


#############
# 0 < x < 1 #
#############

const UnitDistribution = Union{Beta, KSOneSided, NoncentralBeta}

link(d::UnitDistribution, x::Real) = StatsFuns.logit(x)
invlink(d::UnitDistribution, y::Real) = StatsFuns.logistic(y)
link(d::UnitDistribution, x::AbstractArray{<:Real}) = StatsFuns.logit.(x)
invlink(d::UnitDistribution, y::AbstractArray{<:Real}) = StatsFuns.logistic.(y)
function logpdf_with_trans(d::UnitDistribution, x::Real, transform::Bool)
    return logpdf(d, x) + transform * log(x * (one(x) - x))
end


###########
# ∑xᵢ = 1 #
###########

const SimplexDistribution = Union{Dirichlet}

function link(
    d::SimplexDistribution,
    x::AbstractVecOrMat{<:Real},
    proj::Bool = true,
)
    return SimplexBijector{proj}()(x)
end

function link_jacobian(
    d::SimplexDistribution,
    x::AbstractVector{T},
    proj::Bool = true,
) where {T<:Real}
    return jacobian(SimplexBijector{proj}(), x)
end

function invlink(
    d::SimplexDistribution,
    y::AbstractVecOrMat{<:Real},
    proj::Bool = true
)
    return inv(SimplexBijector{proj}())(y)
end

function invlink_jacobian(
    d::SimplexDistribution,
    y::AbstractVector{T},
    proj::Bool = true
) where {T<:Real}
    return jacobian(inv(SimplexBijector{proj}()), y)
end

function logpdf_with_trans(
    d::SimplexDistribution,
    x::AbstractVecOrMat{<:Real},
    transform::Bool,
)
    ϵ = _eps(eltype(x))
    lp = logpdf(d, x .+ ϵ)
    if transform
        lp -= logabsdetjac(bijector(d), x)
    end
    return lp
end

###############
# MvLogNormal #
###############

using Distributions: AbstractMvLogNormal

link(d::AbstractMvLogNormal, x::AbstractVecOrMat{<:Real}) = log.(x)
invlink(d::AbstractMvLogNormal, y::AbstractVecOrMat{<:Real}) = exp.(y)
function logpdf_with_trans(
    d::AbstractMvLogNormal,
    x::AbstractVector{<:Real},
    transform::Bool,
)
    if transform
        return logpdf(d, x) - logabsdetjac(Log{1}(), x)
    else
        return logpdf(d, x)
    end
end
function logpdf_with_trans(
    d::AbstractMvLogNormal,
    x::AbstractMatrix{<:Real},
    transform::Bool,
)
    if transform
        return logpdf(d, x) .- logabsdetjac(Log{1}(), x)
    else
        return logpdf(d, x)
    end
end

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{InverseWishart, Wishart}

link(d::PDMatDistribution, X::AbstractMatrix{<:Real}) = PDBijector()(X)
invlink(d::PDMatDistribution, Y::AbstractMatrix{<:Real}) = inv(PDBijector())(Y)

function logpdf_with_trans(
    d::PDMatDistribution,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    _logpdf_with_trans_pd(d, X, transform)
end
function logpdf_with_trans(
    d::PDMatDistribution,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    map(x -> _logpdf_with_trans_pd(d, x, transform), X)
end
function _logpdf_with_trans_pd(
    d,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    T = eltype(X)
    Xcf = cholesky(X, check = false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        U = Xcf.U
        lp += sum((dim(d) .- (1:dim(d)) .+ 2) .* log.(diag(U)))
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

function assert_disc_or_unbounded(d)
    @assert d isa DiscreteUnivariateDistribution || minimum(d) == -Inf && maximum(d) == Inf
end

function link(d::UnivariateDistribution, x::Real)
    assert_disc_or_unbounded(d)
    return x
end
function link(d::UnivariateDistribution, x::AbstractArray{<:Real})
    assert_disc_or_unbounded(d)
    copy.(x)
end

function invlink(d::UnivariateDistribution, y::Real)
    assert_disc_or_unbounded(d)
    return y
end
function invlink(d::UnivariateDistribution, y::AbstractArray{<:Real})
    assert_disc_or_unbounded(d)
    return copy(y)
end

# MultivariateDistributions
using Distributions: MultivariateDistribution

link(d::MultivariateDistribution, x::AbstractVecOrMat{<:Real}) = copy(x)

invlink(d::MultivariateDistribution, y::AbstractVecOrMat{<:Real}) = copy(y)

function logpdf_with_trans(d::MultivariateDistribution, x::AbstractVecOrMat{<:Real}, ::Bool)
    return logpdf(d, x)
end

# MatrixDistributions
using Distributions: MatrixDistribution

link(d::MatrixDistribution, X::AbstractMatrix{<:Real}) = copy(X)
link(d::MatrixDistribution, X::AbstractArray{<:AbstractMatrix{<:Real}}) = map(X) do x
    link(d, x)
end

invlink(d::MatrixDistribution, Y::AbstractMatrix{<:Real}) = copy(Y)
function invlink(d::MatrixDistribution, Y::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(Y) do y
        invlink(d, y)
    end
end

function logpdf_with_trans(
    d::MatrixDistribution,
    X::Union{AbstractMatrix{<:Real}, AbstractArray{<:AbstractMatrix{<:Real}}},
    ::Bool,
)
    return logpdf(d, X)
end

include("flatten.jl")
include("interface.jl")

# optional dependencies
function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("compat/forwarddiff.jl")
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("compat/tracker.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("compat/zygote.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("compat/reversediff.jl")
    @require DistributionsAD="ced4e74d-a319-5a8a-b0ac-84af2272839c" include("compat/distributionsad.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("compat/flux.jl")
end

end # module
