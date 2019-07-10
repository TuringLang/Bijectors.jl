using Distributions, Bijectors
using ForwardDiff
using Tracker

using Turing

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf


abstract type Bijector end
abstract type ADBijector{AD} <: Bijector end

struct Inversed{B <: Bijector} <: Bijector
    orig::B
end

Broadcast.broadcastable(b::Bijector) = Ref(b)

logabsdetjac(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`logabsdetjac(b::$T1, y::$T2)` is not implemented.")
forward(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`forward(b::$T1, y::$T2)` is not implemented.")
transform(b::T1, y::T2) where {T<:Bijector,T1<:Inversed{T},T2} = 
    error("`transform(b::$T1, y::$T2)` is not implemented.")


transform(b::Bijector) = x -> transform(b, x)
forward(ib::Inversed{<: Bijector}, y) = (transform(ib, y), logabsdetjac(ib, y))
logabsdetjac(ib::Inversed{<: Bijector}, y) = - logabsdetjac(ib.orig, transform(ib, y))

Base.inv(b::Bijector) = Inversed(b)
Base.inv(ib::Inversed{<:Bijector}) = ib.orig


# TODO: rename? a bit of a mouthful
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

# Example bijector
struct Identity <: Bijector end
transform(::Identity, x) = x
transform(::Inversed{Identity}, y) = y
forward(::Identity, x) = (x, zero(x))
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
