using Distributions, Bijectors
using ForwardDiff
using Tracker

import Base: inv, ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf

#######################################
# AD stuff "extracted" from Turing.jl #
#######################################

abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct TrackerAD <: ADBackend end

const ADBACKEND = Ref(:forward_diff)
function setadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
    backend_sym == :forward_diff
    ADBACKEND[] = backend_sym
end

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))
function ADBackend(::Val{T}) where {T}
    if T === :forward_diff
        return ForwardDiffAD
    else
        return TrackerAD
    end
end

######################
# Bijector interface #
######################

"Abstract type for a `Bijector`."
abstract type Bijector end

Broadcast.broadcastable(b::Bijector) = Ref(b)

"Abstract type for a `Bijector` making use of auto-differentation (AD)."
abstract type ADBijector{AD} <: Bijector end

"""
    inv(b::Bijector)
    Inversed(b::Bijector)

A `Bijector` representing the inverse transform of `b`.
"""
struct Inversed{B <: Bijector} <: Bijector
    orig::B
end

inv(b::Bijector) = Inversed(b)
inv(ib::Inversed{<:Bijector}) = ib.orig

"""
    logabsdetjac(b::Bijector, x)
    logabsdetjac(ib::Inversed{<: Bijector}, y)

Computes the log(abs(det(J(x)))) where J is the jacobian of the transform.
Similarily for the inverse-transform.

Default implementation for `Inversed{<: Bijector}` is implemented as
`- logabsdetjac` of original `Bijector`.
"""
logabsdetjac(ib::Inversed{<: Bijector}, y) = - logabsdetjac(ib.orig, ib(y))

"""
    forward(b::Bijector, x)
    forward(ib::Inversed{<: Bijector}, y)

Computes both `transform` and `logabsdetjac` in one forward pass, and
returns a named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`.

This defaults to the call above, but often one can re-use computation
in the computation of the forward pass and the computation of the
`logabsdetjac`. `forward` allows the user to take advantange of such
efficiencies, if they exist.
"""
forward(b::Bijector, x) = (rv=b(x), logabsdetjac=logabsdetjac(b, x))
forward(ib::Inversed{<: Bijector}, y) = (
    rv=ib(y),
    logabsdetjac=logabsdetjac(ib, y)
)


# AD implementations
function jacobian(b::ADBijector{<: ForwardDiffAD}, x::Real)
    return ForwardDiff.derivative(b, x)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::Real)
    return ForwardDiff.derivative(b, y)
end
function jacobian(b::ADBijector{<: ForwardDiffAD}, x::AbstractVector{<: Real})
    return ForwardDiff.jacobian(b, x)
end
function jacobian(b::Inversed{<: ADBijector{<: ForwardDiffAD}}, y::AbstractVector{<: Real})
    return ForwardDiff.jacobian(b, y)
end

function jacobian(b::ADBijector{<: TrackerAD}, x::Real)
    return Tracker.gradient(b, x)[1]
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::Real)
    return Tracker.gradient(b, y)[1]
end
function jacobian(b::ADBijector{<: TrackerAD}, x::AbstractVector{<: Real})
    # we extract `data` so that we don't returne a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, x))
end
function jacobian(b::Inversed{<: ADBijector{<: TrackerAD}}, y::AbstractVector{<: Real})
    # we extract `data` so that we don't returne a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, y))
end

# TODO: allow batch-computation, especially for univariate case?
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
logabsdetjac(b::ADBijector, x::Real) = log(abs(jacobian(b, x)))
function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? log(abs(det(fact))) : -Inf # TODO: do this or not?
end

"""
    logabsdetjacinv(b::Bijector, y)

Just an alias for `logabsdetjac(inv(b), y)`.
"""
logabsdetjacinv(b::Bijector, y) = logabsdetjac(inv(b), y)

###############
# Composition #
###############

"""
    ∘(b1::Bijector, b2::Bijector)
    compose(ts::Bijector...)

A `Bijector` representing composition of bijectors.

# Examples
It's important to note that `∘` does what is expected mathematically, which means that the
bijectors are applied to the input right-to-left, e.g. first applying `b2` and then `b1`:
```
(b1 ∘ b2)(x) == b1(b2(x))     # => true
```
But in the `Composed` struct itself, we store the bijectors left-to-right, so that
```
cb1 = b1 ∘ b2                  # => Composed.ts == [b2, b1]
cb2 = compose(b2, b1)
cb1(x) == cb2(x) == b1(b2(x))  # => true
```
"""
struct Composed{A} <: Bijector
    ts::A
end

compose(ts::Bijector...) = Composed(ts)

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
∘(b1::Bijector, b2::Bijector) = compose(b2, b1)

inv(ct::Composed) = Composed(map(inv, reverse(ct.ts)))

# # TODO: should arrays also be using recursive implementation instead?
function (cb::Composed{<: AbstractArray{<: Bijector}})(x)
    res = x
    for b ∈ cb.ts
        res = b(res)
    end

    return res
end

# recursive implementation like this allows type-inference
_transform(x, b1::Bijector, b2::Bijector) = b2(b1(x))
_transform(x, b::Bijector, bs::Bijector...) = _transform(b(x), bs...)
(cb::Composed{<: Tuple})(x) = _transform(x, cb.ts...)

function _logabsdetjac(x, b1::Bijector, b2::Bijector)
    res = forward(b1, x)
    return logabsdetjac(b2, res.rv) + res.logabsdetjac
end
function _logabsdetjac(x, b1::Bijector, bs::Bijector...)
    res = forward(b1, x)
    return _logabsdetjac(res.rv, bs...) + res.logabsdetjac
end
logabsdetjac(cb::Composed, x) = _logabsdetjac(x, cb.ts...)

# TODO: implement `forward` recursively
function forward(cb::Composed, x)
    res = (rv=x, logabsdetjac=0)
    for t in cb.ts
        res′ = forward(t, res.rv)
        res = (rv=res′.rv, logabsdetjac=res.logabsdetjac + res′.logabsdetjac)
    end
    return res
end

##############################
# Example bijector: Identity #
##############################

struct Identity <: Bijector end
(::Identity)(x) = x
(::Inversed{Identity})(y) = y

forward(::Identity, x) = (rv=x, logabsdetjac=zero(x))

logabsdetjac(::Identity, y::T) where T <: Real = zero(T)
logabsdetjac(::Identity, y::AbstractVector{T}) where T <: Real = zero(T)

const IdentityBijector = Identity()

###############################
# Example: Logit and Logistic #
###############################
using StatsFuns: logit, logistic

struct Logit{T<:Real} <: Bijector
    a::T
    b::T
end

(b::Logit)(x) = @. logit((x - b.a) / (b.b - b.a))
(ib::Inversed{<: Logit{<: Real}})(y) = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a

logabsdetjac(b::Logit{<:Real}, x) = log((x - b.a) * (b.b - x) / (b.b - b.a))

#############
# Exp & Log #
#############

struct Exp <: Bijector end
struct Log <: Bijector end
const exp_b = Exp()
const log_b = Log()

(b::Log)(x) = @. log(x)
(b::Exp)(y) = @. exp(y)

inv(b::Log) = exp_b
inv(b::Exp) = log_b

logabsdetjac(b::Log, x) = log(x)
logabsdetjac(b::Exp, y) = - y

#################
# Shift & Scale #
#################
struct Shift{T} <: Bijector
    a::T
end

(b::Shift)(x) = b.a + x
inv(b::Shift) = Shift(-b.a)
logabsdetjac(b::Shift, x::T) where T = zero(T)

struct Scale{T} <: Bijector
    a::T
end

(b::Scale)(x) = b.a * x
inv(b::Scale) = Scale(b^(-1))
logabsdetjac(b::Scale, x) = log(abs(b.a))

#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
"""
    DistributionBijector(d::Distribution)
    DistributionBijector{<: ADBackend, D}(d::Distribution)

This is the default `Bijector` for a distribution. 

It uses `link` and `invlink` to compute the transformations, and `AD` to compute
the `jacobian` and `logabsdetjac`.
"""
struct DistributionBijector{AD, D} <: ADBijector{AD} where D <: Distribution
    dist::D
end
function DistributionBijector(dist::D) where D <: Distribution
    DistributionBijector{ADBackend(), D}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
(b::DistributionBijector)(x) = link(b.dist, x)
(ib::Inversed{<: DistributionBijector})(y) = invlink(ib.orig.dist, y)

"Returns the constrained-to-unconstrained bijector for distribution `d`."
bijector(d::Distribution) = DistributionBijector(d)

# Transformed distributions
struct TransformedDistribution{D, B, V} <: Distribution{V, Continuous} where {D <: Distribution{V, Continuous}, B <: Bijector}
    dist::D
    transform::B
end
function TransformedDistribution(d::D, b::B) where {V <: VariateForm, B <: Bijector, D <: Distribution{V, Continuous}}
    return TransformedDistribution{D, B, V}(d, b)
end


const UnivariateTransformed = TransformedDistribution{<: Distribution, <: Bijector, Univariate}
const MultivariateTransformed = TransformedDistribution{<: Distribution, <: Bijector, Multivariate}
const MatrixTransformed = TransformedDistribution{<: Distribution, <: Bijector, Matrixvariate}
const Transformed = Union{UnivariateTransformed, MultivariateTransformed, MatrixTransformed}


"""
    transformed(d::Distribution)
    transformed(d::Distribution, b::Bijector)

Couples distribution `d` with the bijector `b` by returning a `TransformedDistribution`.

If no bijector is provided, i.e. `transformed(d)` is called, then 
`transformed(d, bijector(d))` is returned.
"""
transformed(d::Distribution, b::Bijector) = TransformedDistribution(d, b)
transformed(d) = transformed(d, bijector(d))

"""
    bijector(d::Distribution)

Returns the constrained-to-unconstrained bijector for distribution `d`.
"""
bijector(d::Normal) = IdentityBijector
bijector(d::MvNormal) = IdentityBijector
bijector(d::PositiveDistribution) = log_b

_union2tuple(T1::Type, T2::Type) = (T1, T2)
_union2tuple(T1::Type, T2::Union) = (T1, _union2tuple(T2.a, T2.b)...)
_union2tuple(T::Union) = _union2tuple(T.a, T.b)

bijector(d::Kolmogorov) = Logit(zero(eltype(d)), zero(eltype(d)))
for D in _union2tuple(UnitDistribution)[2:end]
    # skipping Kolmogorov because it's a DataType
    @eval bijector(d::$D{T}) where T <: Real = Logit(zero(T), one(T))
end

function bijector(d::Truncated{D}) where D <: Distribution
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return Logit(a, b)
    elseif lowerbounded
        return (log_b ∘ Shift(- a))
    else
        return (log_b ∘ Shift(b) ∘ Scale(- one(typeof(b))))
    end
end

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::Transformed) = length(td.dist)
Base.size(td::Transformed) = size(td.dist)

# TODO: should eventually drop using `logpdf_with_trans` and replace with
# res = forward(inv(td.transform), y)
# logpdf(td.dist, res.rv) .- res.logabsdetjac
function logpdf(td::UnivariateTransformed, y::Real)
    return logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end
function _logpdf(td::MultivariateTransformed, y::AbstractVector{<: Real})
    return logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end

function _logpdf(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    return logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end

function logpdf_with_jac(td::UnivariateTransformed, y::Real)
    res = forward(inv(td.transform), y)
    return (logpdf_with_trans(td.dist, res.rv, true), res.logabsdetjac)
end

function logpdf_with_jac(td::MultivariateTransformed, y::AbstractVector{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf_with_trans(td.dist, res.rv, true), res.logabsdetjac)
end

function logpdf_with_jac(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf_with_trans(td.dist, res.rv, true), res.logabsdetjac)
end

# rand
rand(td::UnivariateTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::UnivariateTransformed) = td.transform(rand(rng, td.dist))

# These ovarloadings are useful for differentiating sampling wrt. params of `td.dist`
# or params of `Bijector`, as they are not inplace like the default `rand`
rand(td::MultivariateTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::MultivariateTransformed) = td.transform(rand(rng, td.dist))
function rand(rng::AbstractRNG, td::MultivariateTransformed, num_samples::Int)
    res = hcat([td.transform(rand(td.dist)) for i = 1:num_samples]...)
    return res
end

function _rand!(rng::AbstractRNG, td::MultivariateTransformed, x::AbstractVector{<: Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end

function _rand!(rng::AbstractRNG, td::MatrixTransformed, x::DenseMatrix{<: Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end
