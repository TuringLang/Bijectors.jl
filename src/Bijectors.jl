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

using Reexport
@reexport using Distributions
using LinearAlgebra
using MappedArrays
using Base.Iterators: drop
using LinearAlgebra: AbstractTriangular, Hermitian

using InverseFunctions: InverseFunctions

import ChangesOfVariables: ChangesOfVariables, with_logabsdet_jacobian
import InverseFunctions: inverse

using ChainRulesCore: ChainRulesCore
using Functors: Functors
using IrrationalConstants: IrrationalConstants
using LogExpFunctions: LogExpFunctions
using Roots: Roots
using DocStringExtensions: TYPEDFIELDS

export TransformDistribution,
    PositiveDistribution,
    UnitDistribution,
    SimplexDistribution,
    PDMatDistribution,
    link,
    invlink,
    logpdf_with_trans,
    isclosedform,
    transform,
    transform!,
    with_logabsdet_jacobian,
    with_logabsdet_jacobian!,
    inverse,
    logabsdetjac,
    logabsdetjac!,
    logabsdetjacinv,
    Bijector,
    Inverse,
    Stacked,
    bijector,
    transformed,
    UnivariateTransformed,
    MultivariateTransformed,
    PlanarLayer,
    RadialLayer,
    Coupling,
    InvertibleBatchNorm,
    elementwise

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_BIJECTORS", "0")))
_debug(str) = @debug str

_eps(::Type{T}) where {T} = T(eps(T))
_eps(::Type{Real}) = eps(Float64)
_eps(::Type{<:Integer}) = eps(Float64)

function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    DEBUG && _debug("x = $x, bounds = $((a, b)), clamped_x = $clamped_x")
    return clamped_x
end

function mapvcat(f, args...)
    out = map(f, args...)
    init = vcat(out[1])
    return reduce(vcat, drop(out, 1); init=init)
end

function maphcat(f, args...)
    out = map(f, args...)
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init=init)
end
function eachcolmaphcat(f, x1, x2)
    out = [f(x1[:, i], x2[i]) for i in 1:size(x1, 2)]
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init=init)
end
function eachcolmaphcat(f, x)
    out = map(f, eachcol(x))
    init = reshape(out[1], :, 1)
    return reduce(hcat, drop(out, 1); init=init)
end
function sumeachcol(f, x1, x2)
    # Using a view below for x1 breaks Tracker
    return sum(f(x1[:, i], x2[i]) for i in 1:size(x1, 2))
end

# Distributions

"""
    link(d::Distribution, x)

Transforms the input `x` using the constrained-to-unconstrained bijector for
distribution `d`.

See also: [`invlink`](@ref).

# Example

```jldoctest
julia> using Bijectors

julia> d = LogNormal()   # support is (0, Inf)
LogNormal{Float64}(μ=0.0, σ=1.0)

julia> b = bijector(d)   # log function transforms to unconstrained space
(::Base.Fix1{typeof(broadcast), typeof(log)}) (generic function with 3 methods)

julia> b(1.0)
0.0

julia> link(LogNormal(), 1.0)
0.0
```
"""
link(d::Distribution, x) = bijector(d)(x)

"""
    invlink(d::Distribution, y)

Performs the inverse transform on a value `y` that was transformed using the
constrained-to-unconstrained bijector for distribution `d`.

It should hold that `invlink(d, link(d, x)) = x`.

See also: [`link`](@ref).

# Example

```jldoctest
julia> using Bijectors

julia> d = LogNormal()    # support is (0, Inf)
LogNormal{Float64}(μ=0.0, σ=1.0)

julia> link(LogNormal(), 1.0)   # uses a log transform
0.0

julia> invlink(LogNormal(), 0.0)
1.0
```
"""
invlink(d::Distribution, y) = inverse(bijector(d))(y)

# To still allow `logpdf_with_trans` to work with "batches" in a similar way
# as `logpdf` can.

# Default catch-all so we can work with distributions by default and batch-support can be
# added when needed.
_logabsdetjac_dist(d::Distribution, x) = logabsdetjac(bijector(d), x)

_logabsdetjac_dist(d::UnivariateDistribution, x::Real) = logabsdetjac(bijector(d), x)
function _logabsdetjac_dist(d::UnivariateDistribution, x::AbstractArray)
    return logabsdetjac.((bijector(d),), x)
end

function _logabsdetjac_dist(d::MultivariateDistribution, x::AbstractVector)
    return logabsdetjac(bijector(d), x)
end
function _logabsdetjac_dist(d::MultivariateDistribution, x::AbstractMatrix)
    return map(Base.Fix1(logabsdetjac, bijector(d)), eachcol(x))
end

_logabsdetjac_dist(d::MatrixDistribution, x::AbstractMatrix) = logabsdetjac(bijector(d), x)
function _logabsdetjac_dist(d::MatrixDistribution, x::AbstractVector{<:AbstractMatrix})
    return logabsdetjac.((bijector(d),), x)
end

_logabsdetjac_dist(d::LKJCholesky, x::Cholesky) = logabsdetjac(bijector(d), x)
_logabsdetjac_dist(d::LKJCholesky, x::AbstractVector) = logabsdetjac.((bijector(d),), x)

"""
    logpdf_with_trans(d::Distribution, x, transform::Bool)

If `transform` is `false`, `logpdf_with_trans` calculates the log probability
density function (logpdf) of distribution `d` at `x`.

If `transform` is `true`, `x` is transformed using the
constrained-to-unconstrained bijector for distribution `d`, and then the logpdf
of the resulting value is calculated with respect to the unconstrained
(transformed) distribution. Equivalently, if `x` is distributed according to
`d` and `y = link(d, x)` is distributed according to `td = transformed(d)`,
then `logpdf_with_trans(d, x, true) = logpdf(td, y)`. This is accomplished
by subtracting the log Jacobian of the transformation.

# Example

```jldoctest
julia> using Bijectors

julia> logpdf_with_trans(LogNormal(), ℯ, false)
-2.4189385332046727

julia> logpdf(LogNormal(), ℯ)  # Same as above
-2.4189385332046727

julia> logpdf_with_trans(LogNormal(), ℯ, true)
-1.4189385332046727

julia> # If x ~ LogNormal(), then log(x) ~ Normal()
       logpdf(Normal(), 1.0)   
-1.4189385332046727

julia> # The difference between the two is due to the Jacobian
       logabsdetjac(bijector(LogNormal()), ℯ)
-1
```
"""
function logpdf_with_trans(d::Distribution, x, transform::Bool)
    if ispd(d)
        return pd_logpdf_with_trans(d, x, transform)
    elseif isdirichlet(d)
        l = logpdf(d, x .+ _eps(eltype(x)))
    else
        l = logpdf(d, x)
    end
    if transform
        return l - _logabsdetjac_dist(d, x)
    else
        return l
    end
end

## Univariate

const TransformDistribution =
    Union{T,Truncated{T}} where {T<:ContinuousUnivariateDistribution}
const PositiveDistribution = Union{
    BetaPrime,
    Chi,
    Chisq,
    Erlang,
    Exponential,
    FDist,
    Frechet,
    Gamma,
    InverseGamma,
    InverseGaussian,
    Kolmogorov,
    LogNormal,
    NoncentralChisq,
    NoncentralF,
    Rayleigh,
    Weibull,
}
const UnitDistribution = Union{Beta,KSOneSided,NoncentralBeta}

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

link(d::Dirichlet, x::AbstractVecOrMat{<:Real}) = SimplexBijector()(x)
link_jacobian(d::Dirichlet, x::AbstractVector{<:Real}) = jacobian(SimplexBijector(), x)

invlink(d::Dirichlet, y::AbstractVecOrMat{<:Real}) = inverse(SimplexBijector())(y)
function invlink_jacobian(d::Dirichlet, y::AbstractVector{<:Real})
    return jacobian(inverse(SimplexBijector()), y)
end

## Matrix

#####################
# Positive definite #
#####################

const PDMatDistribution = Union{MatrixBeta,InverseWishart,Wishart}
ispd(::Distribution) = false
ispd(::PDMatDistribution) = true

function logpdf_with_trans(
    d::MatrixDistribution, X::AbstractArray{<:AbstractMatrix{<:Real}}, transform::Bool
)
    return map(X) do x
        logpdf_with_trans(d, x, transform)
    end
end
function pd_logpdf_with_trans(d, X::AbstractMatrix{<:Real}, transform::Bool)
    T = eltype(X)
    Xcf = cholesky(X; check=false)
    if !issuccess(Xcf)
        Xcf = cholesky(X + max(eps(T), eps(T) * norm(X)) * I)
    end
    lp = getlogp(d, Xcf, X)
    if transform && isfinite(lp)
        n = size(d, 1)
        lp += sum(((n + 2) .- (1:n)) .* log.(diag(Xcf.factors)))
        lp += n * oftype(lp, IrrationalConstants.logtwo)
    end
    return lp
end
function getlogp(d::MatrixBeta, Xcf, X)
    n1, n2 = params(d)
    p = size(d, 1)
    return ((n1 - p - 1) / 2) * logdet(Xcf) + ((n2 - p - 1) / 2) * logdet(I - X) + d.logc0
end
function getlogp(d::Wishart, Xcf, X)
    return ((d.df - (size(d, 1) + 1)) * logdet(Xcf) - tr(d.S \ X)) / 2 + d.logc0
end
function getlogp(d::InverseWishart, Xcf, X)
    Ψ = Matrix(d.Ψ)
    return -((d.df + size(d, 1) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) / 2 + d.logc0
end

include("utils.jl")
include("interface.jl")
include("chainrules.jl")

# Broadcasting here breaks Tracker for some reason
maporbroadcast(f, x::AbstractArray{<:Any,N}...) where {N} = map(f, x...)
maporbroadcast(f, x::AbstractArray...) = f.(x...)

end # module
