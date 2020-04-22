module Bijectors

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

function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    ϵ = _eps(T)
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    DEBUG && _debug("x = $x, bounds = $((a, b)), clamped_x = $clamped_x")
    return clamped_x
end

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

# Distributions

link(d::Distribution, x) = bijector(d)(x)
invlink(d::Distribution, y) = inv(bijector(d))(y)
function logpdf_with_trans(d::Distribution, x, transform::Bool)
    if transform
        return logpdf(d, x) - logabsdetjac(bijector(d), x)
    else
        return logpdf(d, x)
    end
end

## Univariate

const TransformDistribution = Union{
    T,
    Truncated{T},
} where T <: ContinuousUnivariateDistribution
const PositiveDistribution = Union{
    BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma,
    InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull,
}
const UnitDistribution = Union{Beta, KSOneSided, NoncentralBeta}

function logpdf_with_trans(d::UnivariateDistribution, x::AbstractArray{<:Real}, trans::Bool)
    if toflatten(d)
        f, args = flatten(d, trans)
        return f.(args..., x)
    else
        return map(x) do x
            logpdf_with_trans(d, x, trans)
        end
    end
end

function logpdf_with_trans(d::DiscreteUnivariateDistribution, x::Integer, transform::Bool)
    return logpdf(d, x)
end

## Multivariate

function logpdf_with_trans(
    dist::Distributions.Product,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return sum(maporbroadcast(dist.v, x) do d, x
        logpdf_with_trans(d, x, istrans)
    end)
end
function logpdf_with_trans(
    dist::Distributions.Product,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return map(eachcol(x)) do x
        logpdf_with_trans(dist, x, istrans)
    end
end

link(dist::Distributions.Product{Discrete}, x::AbstractVector{<:Real}) = copy(x)
function link(
    dist::Distributions.Product{Continuous},
    x::AbstractVector{<:Real},
)
    return maporbroadcast(link, dist.v, x)
end

link(dist::Distributions.Product{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function link(
    dist::Distributions.Product{Continuous},
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x) do c
        link(dist, c)
    end
end

invlink(dist::Distributions.Product{Discrete}, x::AbstractVector{<:Real}) = copy(x)
function invlink(
    dist::Distributions.Product{Continuous},
    x::AbstractVector{<:Real},
)
    return maporbroadcast(invlink, dist.v, x)
end

invlink(dist::Distributions.Product{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function invlink(
    dist::Distributions.Product{Continuous},
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x) do c
        invlink(dist, c)
    end
end

function maporbroadcast(f, dists::AbstractArray, x::AbstractArray)
    # Broadcasting here breaks Tracker for some reason
    return map(f, dists, x)
end
function maporbroadcast(f, dists::AbstractVector, x::AbstractMatrix)
    return map(x -> sum(maporbroadcast(f, dists, x)), eachcol(x))
end

const SimplexDistribution = Union{Dirichlet}

function logpdf_with_trans(
    d::DiscreteMultivariateDistribution,
    x::AbstractVecOrMat{<:Real},
    ::Bool,
)
    return logpdf(d, x)
end

###########
# ∑xᵢ = 1 #
###########

function link(
    d::Dirichlet,
    x::AbstractVecOrMat{<:Real},
    proj::Bool = true,
)
    return SimplexBijector{proj}()(x)
end

function link_jacobian(
    d::Dirichlet,
    x::AbstractVector{T},
    proj::Bool = true,
) where {T<:Real}
    return jacobian(SimplexBijector{proj}(), x)
end

function invlink(
    d::Dirichlet,
    y::AbstractVecOrMat{<:Real},
    proj::Bool = true
)
    return inv(SimplexBijector{proj}())(y)
end

function invlink_jacobian(
    d::Dirichlet,
    y::AbstractVector{T},
    proj::Bool = true
) where {T<:Real}
    return jacobian(inv(SimplexBijector{proj}()), y)
end

function logpdf_with_trans(
    d::Dirichlet,
    x::AbstractVecOrMat{<:Real},
    transform::Bool,
)
    return dirichlet_logpdf_with_trans(d, x, transform)
end
function dirichlet_logpdf_with_trans(d, x, transform)
    ϵ = _eps(eltype(x))
    lp = logpdf(d, x .+ ϵ)
    if transform
        lp -= logabsdetjac(bijector(d), x)
    end
    return lp
end

## Matrix

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{InverseWishart, Wishart}

function logpdf_with_trans(
    d::PDMatDistribution,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool,
)
    return map(X) do x
        logpdf_with_trans(d, x, transform)
    end
end
function logpdf_with_trans(
    d::PDMatDistribution,
    X::AbstractMatrix{<:Real},
    transform::Bool,
)
    pd_logpdf_with_trans(d, X, transform)
end
function pd_logpdf_with_trans(
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

include("flatten.jl")
include("interface.jl")

# optional dependencies
function __init__()
    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        function maporbroadcast(f, dists::LazyArrays.BroadcastArray, x::AbstractArray)
            return copy(f.(dists, x))
        end
        function maporbroadcast(f, dists::LazyArrays.BroadcastVector, x::AbstractMatrix)
            return vec(sum(copy(f.(dists, x)), dims = 1))
        end
    end
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("compat/forwarddiff.jl")
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("compat/tracker.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("compat/zygote.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("compat/reversediff.jl")
    @require DistributionsAD="ced4e74d-a319-5a8a-b0ac-84af2272839c" include("compat/distributionsad.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("compat/flux.jl")
end

end # module
