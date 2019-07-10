using Distributions, Bijectors
using ForwardDiff
using Tracker

using Turing

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf

import Base: inv


abstract type Bijector end
abstract type ADBijector{AD} <: Bijector end

struct Inversed{B <: Bijector} <: Bijector
    orig::B
end

Broadcast.broadcastable(b::Bijector) = Ref(b)

"Computes the log(abs(det(J(x)))) where J is the jacobian of the transform."
logabsdetjac(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`logabsdetjac(b::$T1, y::$T2)` is not implemented.")

"Transforms the input using the bijector."
transform(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} =
    error("`transform(b::$T1, y::$T2)` is not implemented.")

"Computes both `transform` and `logabsdetjac` in one forward pass."
forward(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`forward(b::$T1, y::$T2)` is not implemented.")


transform(b::Bijector) = x -> transform(b, x)

# default `forward` implementations; should in general implement efficient way
# of computing both `transform` and `logabsdetjac` together.
forward(b::Bijector, x) = (rv=transform(b, x), logabsdetjac=logabsdetjac(b, x))
forward(ib::Inversed{<: Bijector}, y) = (rv=transform(ib, y), logabsdetjac=logabsdetjac(ib, y))

# defaults implementation for inverses
logabsdetjac(ib::Inversed{<: Bijector}, y) = - logabsdetjac(ib.orig, transform(ib, y))

inv(b::Bijector) = Inversed(b)
inv(ib::Inversed{<:Bijector}) = ib.orig

# AD implementations

# TODO: allow batch-computation, especially for univariate case
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
function logabsdetjac(b::ADBijector{<: Turing.Core.ForwardDiffAD}, y::Real)
    log(abs(ForwardDiff.derivative(z -> transform(b, z), y)))
end
function logabsdetjac(b::ADBijector{<:Turing.Core.ForwardDiffAD}, y::AbstractVector{<:Real})
    logabsdet(ForwardDiff.jacobian(z -> transform(b, z), y))[1]
end

# FIXME: untrack? i.e. `Tracker.data(...)`
function logabsdetjac(b::ADBijector{<: Turing.Core.TrackerAD}, y::Real)
    log(abs(Tracker.gradient(z -> transform(b, z[1]), [y])[1][1]))
end
function logabsdetjac(b::ADBijector{<: Turing.Core.TrackerAD}, y::AbstractVector{<: Real})
    logabsdet(Tracker.jacobian(z -> transform(b, z), y))[1]
end

# Composition

struct Composed{B<:Bijector} <: Bijector
    ts::Vector{B}
end

compose(ts...) = Composed([ts...])

inv(ct::Composed{B}) where {B<:Bijector} = Composed(map(inv, reverse(ct.ts)))

function forward(ct::Composed{<:Bijector}, x)
    res = (rv=x, logabsdetjac=0)
    for t in ct.ts
        res′ = forward(t, res.rv)
        res = (rv=res′.rv, logabsdetjac=res.logabsdetjac + res′.logabsdetjac)
    end
    return res
end

# Example bijector
struct Identity <: Bijector end
transform(::Identity, x) = x
transform(::Inversed{Identity}, y) = y
forward(::Identity, x) = (rv=x, logabsdetjac=zero(x))
logabsdetjac(::Identity, y::T) where T <: Real = zero(T)
logabsdetjac(::Identity, y::AbstractVector{T}) where T <: Real = zero(T)

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
struct DistributionBijector{AD, D} <: ADBijector{AD} where D <: Distribution
    dist::D
end
function DistributionBijector(dist::D) where D <: Distribution
    DistributionBijector{Turing.Core.ADBackend(), D}(dist)
end

transform(b::DistributionBijector, x) = link(b.dist, x)
transform(ib::Inversed{<: DistributionBijector}, y) = invlink(ib.orig.dist, y)

# Transformed distributions
struct UnivariateTransformed{D, B} <: Distribution{Univariate, Continuous} where {D <: UnivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end

struct MultivariateTransformed{D, B} <: Distribution{Multivariate, Continuous} where {D <: MultivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end


# Can implement these on a case-by-case basis
transformed(d::UnivariateDistribution, b::Bijector) = UnivariateTransformed(d, b)
transformed(d::MultivariateDistribution, b::Bijector) = MultivariateTransformed(d, b)

transformed(d) = transformed(d, DistributionBijector(d))

# can specialize further by
transformed(d::Normal) = transformed(d, Identity())

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::MultivariateTransformed) = length(td.dist)

# logp
function logpdf(td::UnivariateTransformed, y::T where T <: Real)
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, transform(inv(td.transform), y), true)
end
function _logpdf(td::MultivariateTransformed, y::AbstractVector{T} where T <: Real)
    # logpdf(td.dist, transform(inv(td.transform), y)) .+ logabsdetjac(inv(td.transform), y)
    logpdf_with_trans(td.dist, transform(inv(td.transform), y), true)
end

# TODO: implement these using analytical expressions?
function logpdf_with_jac(td::UnivariateTransformed, y::T where T <: Real)
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, transform(inv(td.transform), y)) .+ z, z)
end

function logpdf_with_jac(td::MultivariateTransformed, y::AbstractVector{T} where T <: Real)
    z = logabsdetjac(inv(td.transform), y)
    return (logpdf(td.dist, transform(inv(td.transform), y)) .+ z, z)
end

# rand
rand(rng::AbstractRNG, td::UnivariateTransformed) = transform(td.transform, rand(td.dist))
function _rand!(rng::AbstractRNG, td::MultivariateTransformed, x::AbstractVector{<: Real})
    rand!(rng, td.dist, x)
    y = transform(td.transform, x)
    copyto!(x, y)
end
