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
using Base.Iterators: drop
using LinearAlgebra: AbstractTriangular
import NonlinearSolve
import ChainRulesCore

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
        bijector,
        transformed,
        UnivariateTransformed,
        MultivariateTransformed,
        logpdf_with_jac,
        logpdf_forward,
        PlanarLayer,
        RadialLayer,
        CouplingLayer,
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
    if ispd(d)
        return pd_logpdf_with_trans(d, x, transform)
    elseif isdirichlet(d)
        l = logpdf(d, x .+ eps(eltype(x)))
    else
        l = logpdf(d, x)
    end
    if transform
        return l - logabsdetjac(bijector(d), x)
    else
        return l
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

function logpdf_with_trans(d::UnivariateDistribution, x, transform::Bool)
    if transform
        return map(x -> logpdf(d, x), x) - logabsdetjac(bijector(d), x)
    else
        return map(x -> logpdf(d, x), x)
    end
end

## Multivariate

const SimplexDistribution = Union{Dirichlet}
isdirichlet(::SimplexDistribution) = true
isdirichlet(::Distribution) = false

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

## Matrix

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{MatrixBeta, InverseWishart, Wishart}
ispd(::Distribution) = false
ispd(::PDMatDistribution) = true

function logpdf_with_trans(
    d::MatrixDistribution,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool,
)
    return map(X) do x
        logpdf_with_trans(d, x, transform)
    end
end
function pd_logpdf_with_trans(d, X::AbstractMatrix{<:Real}, transform::Bool)
    T = eltype(X)
    Xcf = cholesky(X, check = false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        U = Xcf.U
        d = dim(d)
        lp += sum((d .- (1:d) .+ 2) .* log.(diag(U)))
        lp += d * log(T(2))
    end
    return lp
end
function getlogp(d::MatrixBeta, Xcf, X)
    n1, n2 = params(d)
    p = dim(d)
    return ((n1 - p - 1) / 2) * logdet(Xcf) + ((n2 - p - 1) / 2) * logdet(I - X) + d.logc0
end
function getlogp(d::Wishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.S \ X)) + d.logc0
end
function getlogp(d::InverseWishart, Xcf, X)
    Ψ = Matrix(d.Ψ)
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) + d.logc0
end

include("utils.jl")
include("interface.jl")
include("chainrules.jl")

# Broadcasting here breaks Tracker for some reason
maporbroadcast(f, x::AbstractArray{<:Any, N}...) where {N} = map(f, x...)
maporbroadcast(f, x::AbstractArray...) = f.(x...)

# optional dependencies
function __init__()
    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        function maporbroadcast(f, x1::LazyArrays.BroadcastArray, x...)
            return copy(f.(x1, x...))
        end
        function maporbroadcast(f, x1, x2::LazyArrays.BroadcastArray, x...)
            return copy(f.(x1, x2, x...))
        end
        function maporbroadcast(f, x1, x2, x3::LazyArrays.BroadcastArray, x...)
            return copy(f.(x1, x2, x3, x...))
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
