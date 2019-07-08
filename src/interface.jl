using Distributions, Bijectors
using ForwardDiff
using Tracker

using Turing

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf


abstract type Bijector end
abstract type ADBijector{AD} <: Bijector end

Broadcast.broadcastable(b::Bijector) = Ref(b)

"Computes the transformation."
transform(b::Bijector, x) = begin end
transform(b::Bijector) = x -> transform(b, x)

"Computes the inverse transformation of the Bijector."
inverse(b::Bijector, y) = begin end
inverse(b::Bijector) = y -> inverse(b, y)

# TODO: rename? a bit of a mouthful
# TODO: allow batch-computation, especially for univariate case
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
logdetinvjac(b::Bijector, y) = begin end
logdetinvjac(b::ADBijector{AD}, y::T) where {AD <: Turing.Core.ForwardDiffAD, T <: Real} = log(abs(ForwardDiff.derivative(z -> inverse(b, z), y)))
logdetinvjac(b::ADBijector{AD}, y::AbstractVector) where AD <: Turing.Core.ForwardDiffAD = logabsdet(ForwardDiff.jacobian(z -> inverse(b, z), y))[1]

# FIXME: untrack? i.e. `Tracker.data(...)`
logdetinvjac(b::ADBijector{AD}, y::T) where {AD <: Turing.Core.TrackerAD, T <: Real} = log(abs(Tracker.gradient(z -> inverse(b, z[1]), [y])[1][1]))
logdetinvjac(b::ADBijector{AD}, y::AbstractVector) where AD <: Turing.Core.TrackerAD = logabsdet(Tracker.jacobian(z -> inverse(b, z), y))[1]

# Example bijector
struct Identity <: Bijector end
transform(::Identity, x) = x
inverse(::Identity, y) = y
logdetinvjac(::Identity, y::T) where T <: Real = zero(T)
logdetinvjac(::Identity, y::AbstractVector{T}) where T <: Real = zero(T)

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
struct DistributionBijector{AD, D} <: ADBijector{AD} where D <: Distribution
    dist::D
end
DistributionBijector(dist::D) where D <: Distribution = DistributionBijector{Turing.Core.ADBackend(), D}(dist)

transform(b::DistributionBijector, x) = link(b.dist, x)
inverse(b::DistributionBijector, y) = invlink(b.dist, y)

# Transformed distributions
struct UnivariateTransformed{D, B} <: Distribution{Univariate, Continuous} where {D <: UnivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end

struct MultivariateTransformed{D, B} <: Distribution{Multivariate, Continuous} where {D <: MultivariateDistribution, B <: Bijector}
    dist::D
    transform::B
end


# implement these on a case-by-case basis, e.g. `PDMatDistribution = Union{InverseWishart, Wishart}`
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
    # logpdf(td.dist, inverse(td.transform, y)) .+ logdetinvjac(td.transform, y)
    logpdf_with_trans(td.dist, inverse(td.transform, y), true)
end
function _logpdf(td::MultivariateTransformed, y::AbstractVector{T} where T <: Real)
    # logpdf(td.dist, inverse(td.transform, y)) .+ logdetinvjac(td.transform, y)
    logpdf_with_trans(td.dist, inverse(td.transform, y), true)
end

# TODO: implement these using analytical expressions?
function logpdf_with_jac(td::UnivariateTransformed, y::T where T <: Real)
    z = logdetinvjac(td.transform, y)
    return (logpdf(td.dist, inverse(td.transform, y)) .+ z, z)
end

function logpdf_with_jac(td::MultivariateTransformed, y::AbstractVector{T} where T <: Real)
    z = logdetinvjac(td.transform, y)
    return (logpdf(td.dist, inverse(td.transform, y)) .+ z, z)
end

# rand
rand(rng::AbstractRNG, td::UnivariateTransformed) = transform(td.transform, rand(td.dist))
function _rand!(rng::AbstractRNG, td::MultivariateTransformed, x::AbstractVector{T} where T <: Real)
    rand!(rng, td.dist, x)
    y = transform(td.transform, x)
    copyto!(x, y)
end
