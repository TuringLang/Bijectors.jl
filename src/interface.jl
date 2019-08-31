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

"""
Abstract type for a `Bijector` making use of auto-differentation (AD) to
implement `jacobian` and, by impliciation, `logabsdetjac`.
"""
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
    logabsdetjac(ib::Inversed{<:Bijector}, y)

Computes the log(abs(det(J(x)))) where J is the jacobian of the transform.
Similarily for the inverse-transform.

Default implementation for `Inversed{<:Bijector}` is implemented as
`- logabsdetjac` of original `Bijector`.
"""
logabsdetjac(ib::Inversed{<:Bijector}, y) = - logabsdetjac(ib.orig, ib(y))

"""
    forward(b::Bijector, x)
    forward(ib::Inversed{<:Bijector}, y)

Computes both `transform` and `logabsdetjac` in one forward pass, and
returns a named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`.

This defaults to the call above, but often one can re-use computation
in the computation of the forward pass and the computation of the
`logabsdetjac`. `forward` allows the user to take advantange of such
efficiencies, if they exist.
"""
forward(b::Bijector, x) = (rv=b(x), logabsdetjac=logabsdetjac(b, x))
forward(ib::Inversed{<:Bijector}, y) = (
    rv=ib(y),
    logabsdetjac=logabsdetjac(ib, y)
)


# AD implementations
function jacobian(b::ADBijector{<:ForwardDiffAD}, x::Real)
    return ForwardDiff.derivative(b, x)
end
function jacobian(b::Inversed{<:ADBijector{<:ForwardDiffAD}}, y::Real)
    return ForwardDiff.derivative(b, y)
end
function jacobian(b::ADBijector{<:ForwardDiffAD}, x::AbstractVector{<:Real})
    return ForwardDiff.jacobian(b, x)
end
function jacobian(b::Inversed{<:ADBijector{<:ForwardDiffAD}}, y::AbstractVector{<:Real})
    return ForwardDiff.jacobian(b, y)
end

function jacobian(b::ADBijector{<:TrackerAD}, x::Real)
    return Tracker.gradient(b, x)[1]
end
function jacobian(b::Inversed{<:ADBijector{<:TrackerAD}}, y::Real)
    return Tracker.gradient(b, y)[1]
end
function jacobian(b::ADBijector{<:TrackerAD}, x::AbstractVector{<:Real})
    # We extract `data` so that we don't returne a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, x))
end
function jacobian(b::Inversed{<:ADBijector{<:TrackerAD}}, y::AbstractVector{<:Real})
    # We extract `data` so that we don't returne a `Tracked` type
    return Tracker.data(Tracker.jacobian(b, y))
end

struct SingularJacobianException{B} <: Exception where {B<:Bijector}
    b::B
end
Base.showerror(io::IO, e::SingularJacobianException) = print(io, "jacobian of $(e.b) is singular")

# TODO: allow batch-computation, especially for univariate case?
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
function logabsdetjac(b::ADBijector, x::Real)
    res = log(abs(jacobian(b, x)))
    return isfinite(res) ? res : throw(SingularJacobianException(b))
end

function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? log(abs(det(fact))) : throw(SingularJacobianException(b))
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
    Composed(ts::A)

    ∘(b1::Bijector, b2::Bijector)::Composed{<:Tuple}
    composel(ts::Bijector...)::Composed{<:Tuple}
    composer(ts::Bijector...)::Composed{<:Tuple}

A `Bijector` representing composition of bijectors. `composel` and `composer` results in a
`Composed` for which application occurs from left-to-right and right-to-left, respectively.

Note that all the propsed ways of constructing a `Composed` returns a `Tuple` of bijectors.
This ensures type-stability of implementations of all relating methdos, e.g. `inv`.

# Examples
It's important to note that `∘` does what is expected mathematically, which means that the
bijectors are applied to the input right-to-left, e.g. first applying `b2` and then `b1`:
```
(b1 ∘ b2)(x) == b1(b2(x))     # => true
```
But in the `Composed` struct itself, we store the bijectors left-to-right, so that
```
cb1 = b1 ∘ b2                  # => Composed.ts == (b2, b1)
cb2 = composel(b2, b1)         # => Composed.ts == (b2, b1)
cb1(x) == cb2(x) == b1(b2(x))  # => true
```
"""
struct Composed{A} <: Bijector
    ts::A
end

"""
    composel(ts::Bijector...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied left-to-right.
"""
composel(ts::Bijector...) = Composed(ts)

"""
    composer(ts::Bijector...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied right-to-left.
"""
composer(ts::Bijector...) = Composed(inv(ts))

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
∘(b1::Bijector, b2::Bijector) = composel(b2, b1)

inv(ct::Composed) = Composed(map(inv, reverse(ct.ts)))

# # TODO: should arrays also be using recursive implementation instead?
function (cb::Composed{<:AbstractArray{<:Bijector}})(x)
    res = x
    for b ∈ cb.ts
        res = b(res)
    end

    return res
end

# recursive implementation like this allows type-inference
_transform(x, b1::Bijector, b2::Bijector) = b2(b1(x))
_transform(x, b::Bijector, bs::Bijector...) = _transform(b(x), bs...)
(cb::Composed{<:Tuple})(x) = _transform(x, cb.ts...)

function _logabsdetjac(x, b1::Bijector, b2::Bijector)
    res = forward(b1, x)
    return logabsdetjac(b2, res.rv) .+ res.logabsdetjac
end
function _logabsdetjac(x, b1::Bijector, bs::Bijector...)
    res = forward(b1, x)
    return _logabsdetjac(res.rv, bs...) .+ res.logabsdetjac
end
logabsdetjac(cb::Composed, x) = _logabsdetjac(x, cb.ts...)

# Recursive implementation of `forward`
# NOTE: we need this one in the case where `length(cb.ts) == 2`
# in which case forward(...) immediately calls `_forward(::NamedTuple, b::Bijector)`
function _forward(f::NamedTuple, b::Bijector)
    y, logjac = forward(b, f.rv)
    return (rv=y, logabsdetjac=logjac .+ f.logabsdetjac)
end
function _forward(f::NamedTuple, b1::Bijector, b2::Bijector)
    f1 = forward(b1, f.rv)
    f2 = forward(b2, f1.rv)
    return (rv=f2.rv, logabsdetjac=f2.logabsdetjac .+ f1.logabsdetjac .+ f.logabsdetjac)
end
function _forward(f::NamedTuple, b::Bijector, bs::Bijector...)
    f1 = forward(b, f.rv)
    f_ = (rv=f1.rv, logabsdetjac=f1.logabsdetjac .+ f.logabsdetjac)
    return _forward(f_, bs...)
end
_forward(x, b::Bijector, bs::Bijector...) = _forward(forward(b, x), bs...)
forward(cb::Composed{<:Tuple}, x) = _forward(x, cb.ts...)

function forward(cb::Composed, x)
    rv, logjac = forward(cb.ts[1], x)
    
    for t in cb.ts[2:end]
        res = forward(t, rv)
        rv = res.rv
        logjac = res.logabsdetjac .+ logjac
    end
    return (rv=rv, logabsdetjac=logjac)
end

##############################
# Example bijector: Identity #
##############################

struct Identity <: Bijector end
(::Identity)(x) = x
(::Inversed{Identity})(y) = y

forward(::Identity, x) = (rv=x, logabsdetjac=zero(eltype(x)))

logabsdetjac(::Identity, y) = zero(eltype(y))

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
(ib::Inversed{<:Logit{<:Real}})(y) = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a

logabsdetjac(b::Logit{<:Real}, x) = @. - log((x - b.a) * (b.b - x) / (b.b - b.a))

#############
# Exp & Log #
#############

struct Exp <: Bijector end
struct Log <: Bijector end

(b::Log)(x) = @. log(x)
(b::Exp)(y) = @. exp(y)

inv(b::Log) = Exp()
inv(b::Exp) = Log()

logabsdetjac(b::Log, x) = - sum(log.(x))
logabsdetjac(b::Exp, y) = sum(y)

#################
# Shift & Scale #
#################
struct Shift{T} <: Bijector
    a::T
end

(b::Shift)(x) = b.a + x
(b::Shift{<:Real})(x::AbstractArray) = b.a .+ x
(b::Shift{<:AbstractVector})(x::AbstractMatrix) = b.a .+ x

inv(b::Shift) = Shift(-b.a)
logabsdetjac(b::Shift, x) = zero(eltype(x))
# FIXME: ambiguous whether or not this is actually a batch or whatever
logabsdetjac(b::Shift{<:Real}, x::AbstractMatrix) = zeros(eltype(x), size(x, 2))
logabsdetjac(b::Shift{<:AbstractVector}, x::AbstractMatrix) = zeros(eltype(x), size(x, 2))

struct Scale{T} <: Bijector
    a::T
end

(b::Scale)(x) = b.a * x
(b::Scale{<:Real})(x::AbstractArray) = b.a .* x
(b::Scale{<:AbstractVector{<:Real}})(x::AbstractMatrix{<:Real}) = x * b.a

inv(b::Scale) = Scale(inv(b.a))
inv(b::Scale{<:AbstractVector}) = Scale(inv.(b.a))

# TODO: should this be implemented for batch-computation?
# There's an ambiguity issue
#      logabsdetjac(b::Scale{<: AbstractVector}, x::AbstractMatrix)
# Is this a batch or is it simply a matrix we want to scale differently
# in each component?
logabsdetjac(b::Scale, x) = log(abs(b.a))

####################
# Simplex bijector #
####################
struct SimplexBijector{T} <: Bijector where {T} end

const simplex_b = SimplexBijector{Val{false}}()
const simplex_b_proj = SimplexBijector{Val{true}}()

# The following implementations are basically just copy-paste from `invlink` and
# `link` for `SimplexDistributions` but dropping the dependence on the `Distribution`.
function _clamp(x::T, b::SimplexBijector) where {T}
    bounds = (zero(T), one(T))
    clamped_x = clamp(x, bounds...)
    DEBUG && @debug "x = $x, bounds = $bounds, clamped_x = $clamped_x"
    return clamped_x
end

function (b::SimplexBijector{Val{proj}})(x::AbstractVector{T}) where {T, proj}
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
function (b::SimplexBijector{Val{proj}})(X::AbstractMatrix{T}) where {T<:Real, proj}
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

function (ib::Inversed{<:SimplexBijector{Val{proj}}})(y::AbstractVector{T}) where {T, proj}
    x, K = similar(y), length(y)

    ϵ = _eps(T)
    @inbounds z = StatsFuns.logistic(y[1] - log(T(K - 1)))
    @inbounds x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), ib.orig)
    sum_tmp = zero(T)
    @inbounds @simd for k = 2:(K - 1)
        z = StatsFuns.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, ib.orig)
    end
    @inbounds sum_tmp += x[K - 1]
    @inbounds if proj
        x[K] = _clamp(one(T) - sum_tmp, ib.orig)
    else
        x[K] = _clamp(one(T) - sum_tmp - y[K], ib.orig)
    end
    
    return x
end

# Vectorised implementation of the above.
function (ib::Inversed{<:SimplexBijector{Val{proj}}})(
    Y::AbstractMatrix{T}
) where {T<:Real, proj}
    X, K, N = similar(Y), size(Y, 1), size(Y, 2)

    ϵ = _eps(T)
    @inbounds @simd for n in 1:size(X, 2)
        sum_tmp, z = zero(T), StatsFuns.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), ib.orig)
        for k in 2:(K - 1)
            z = StatsFuns.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k - 1]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, ib.orig)
        end
        sum_tmp += X[K - 1, n]
        if proj
            X[K, n] = _clamp(one(T) - sum_tmp, ib.orig)
        else
            X[K, n] = _clamp(one(T) - sum_tmp - Y[K, n], ib.orig)
        end
    end

    return X
end


function logabsdetjac(b::SimplexBijector, x::AbstractVector{T}) where T
    ϵ = _eps(T)
    lp = zero(T)
    
    K = length(x)

    sum_tmp = zero(eltype(x))
    @inbounds z = x[1]
    lp += log(z + ϵ) + log((one(T) + ϵ) - z)
    @inbounds @simd for k in 2:(K - 1)
        sum_tmp += x[k-1]
        z = x[k] / ((one(T) + ϵ) - sum_tmp)
        lp += log(z + ϵ) + log((one(T) + ϵ) - z) + log((one(T) + ϵ) - sum_tmp)
    end

    return - lp
end

#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
"""
    DistributionBijector(d::Distribution)
    DistributionBijector{<:ADBackend, D}(d::Distribution)

This is the default `Bijector` for a distribution. 

It uses `link` and `invlink` to compute the transformations, and `AD` to compute
the `jacobian` and `logabsdetjac`.
"""
struct DistributionBijector{AD, D} <: ADBijector{AD} where {D<:Distribution}
    dist::D
end
function DistributionBijector(dist::D) where {D<:Distribution}
    DistributionBijector{ADBackend(), D}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
(b::DistributionBijector)(x) = link(b.dist, x)
(ib::Inversed{<:DistributionBijector})(y) = invlink(ib.orig.dist, y)


"Returns the constrained-to-unconstrained bijector for distribution `d`."
bijector(d::Distribution) = DistributionBijector(d)

# Transformed distributions
struct TransformedDistribution{D, B, V} <: Distribution{V, Continuous} where {D<:Distribution{V, Continuous}, B<:Bijector}
    dist::D
    transform::B
end
function TransformedDistribution(d::D, b::B) where {V<:VariateForm, B<:Bijector, D<:Distribution{V, Continuous}}
    return TransformedDistribution{D, B, V}(d, b)
end


const UnivariateTransformed = TransformedDistribution{<:Distribution, <:Bijector, Univariate}
const MultivariateTransformed = TransformedDistribution{<:Distribution, <:Bijector, Multivariate}
const MvTransformed = MultivariateTransformed
const MatrixTransformed = TransformedDistribution{<:Distribution, <:Bijector, Matrixvariate}
const Transformed = TransformedDistribution


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
bijector(d::PositiveDistribution) = Log()
bijector(d::MvLogNormal) = Log()
bijector(d::SimplexDistribution) = simplex_b_proj

_union2tuple(T1::Type, T2::Type) = (T1, T2)
_union2tuple(T1::Type, T2::Union) = (T1, _union2tuple(T2.a, T2.b)...)
_union2tuple(T::Union) = _union2tuple(T.a, T.b)

bijector(d::KSOneSided) = Logit(zero(eltype(d)), zero(eltype(d)))
for D in _union2tuple(UnitDistribution)
    # Skipping KSOneSided because it's not a parametric type
    if D == KSOneSided
        continue
    end
    @eval bijector(d::$D{T}) where {T<:Real} = Logit(zero(T), one(T))
end

# FIXME: (TOR) Can we make this type-stable?
# Everything but `Truncated` can probably be made type-stable
# by explicit implementation. Can also make a `TruncatedBijector`
# which has the same transform as the `link` function.
# E.g. (b::Truncated)(x) = link(b.d, x) or smth
function bijector(d::TransformDistribution) where {D<:Distribution}
    a, b = minimum(d), maximum(d)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return Logit(a, b)
    elseif lowerbounded
        return (Log() ∘ Shift(- a))
    elseif upperbounded
        return (Log() ∘ Shift(b) ∘ Scale(- one(typeof(b))))
    else
        return IdentityBijector
    end
end

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::Transformed) = length(td.dist)
Base.size(td::Transformed) = size(td.dist)

function logpdf(td::UnivariateTransformed, y::Real)
    res = forward(inv(td.transform), y)
    return logpdf(td.dist, res.rv) .+ res.logabsdetjac
end

# TODO: implement more efficiently for flows in the case of `Matrix`
function _logpdf(td::MvTransformed, y::AbstractVector{<:Real})
    res = forward(inv(td.transform), y)
    return logpdf(td.dist, res.rv) .+ res.logabsdetjac
end

function _logpdf(td::MvTransformed{<:Dirichlet}, y::AbstractVector{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    res = forward(inv(td.transform), y)
    return logpdf(td.dist, mappedarray(x->x+ϵ, res.rv)) .+ res.logabsdetjac
end

# TODO: should eventually drop using `logpdf_with_trans` and replace with
# res = forward(inv(td.transform), y)
# logpdf(td.dist, res.rv) .- res.logabsdetjac
function _logpdf(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    return logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end

# rand
rand(td::UnivariateTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::UnivariateTransformed) = td.transform(rand(rng, td.dist))

# These ovarloadings are useful for differentiating sampling wrt. params of `td.dist`
# or params of `Bijector`, as they are not inplace like the default `rand`
rand(td::MvTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::MvTransformed) = td.transform(rand(rng, td.dist))
# TODO: implement more efficiently for flows
function rand(rng::AbstractRNG, td::MvTransformed, num_samples::Int)
    res = hcat([td.transform(rand(td.dist)) for i = 1:num_samples]...)
    return res
end

function _rand!(rng::AbstractRNG, td::MvTransformed, x::AbstractVector{<:Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end

function _rand!(rng::AbstractRNG, td::MatrixTransformed, x::DenseMatrix{<:Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end

#############################################################
# Additional useful functions for `TransformedDistribution` #
#############################################################
"""
    logpdf_with_jac(td::UnivariateTransformed, y::Real)
    logpdf_with_jac(td::MvTransformed, y::AbstractVector{<:Real})
    logpdf_with_jac(td::MatrixTransformed, y::AbstractMatrix{<:Real})

Makes use of the `forward` method to potentially re-use computation
and returns a tuple `(logpdf, logabsdetjac)`.
"""
function logpdf_with_jac(td::UnivariateTransformed, y::Real)
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) .+ res.logabsdetjac, res.logabsdetjac)
end

# TODO: implement more efficiently for flows in the case of `Matrix`
function logpdf_with_jac(td::MvTransformed, y::AbstractVector{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) .+ res.logabsdetjac, res.logabsdetjac)
end

function logpdf_with_jac(td::MvTransformed, y::AbstractMatrix{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) .+ res.logabsdetjac, res.logabsdetjac)
end

function logpdf_with_jac(td::MvTransformed{<:Dirichlet}, y::AbstractVector{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, mappedarray(x->x+ϵ, res.rv)) .+ res.logabsdetjac, res.logabsdetjac)
end

# TODO: should eventually drop using `logpdf_with_trans`
function logpdf_with_jac(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf_with_trans(td.dist, res.rv, true), res.logabsdetjac)
end

"""
    logpdf_forward(td::Transformed, x)
    logpdf_forward(td::Transformed, x, logjac)

Computes the `logpdf` using the forward pass of the bijector rather than using
the inverse transform to compute the necessary `logabsdetjac`.

This is similar to `logpdf_with_trans`.
"""
# TODO: implement more efficiently for flows in the case of `Matrix`
logpdf_forward(td::Transformed, x, logjac) = logpdf(td.dist, x) .- logjac
logpdf_forward(td::Transformed, x) = logpdf_forward(td, x, logabsdetjac(td.transform, x))

function logpdf_forward(td::MvTransformed{<:Dirichlet}, x, logjac)
    T = eltype(x)
    ϵ = _eps(T)

    return logpdf(td.dist, mappedarray(z->z+ϵ, x)) .- logjac
end


# forward function
const GLOBAL_RNG = Distributions.GLOBAL_RNG

function _forward(d::UnivariateDistribution, x)
    y, logjac = forward(IdentityBijector, x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf.(d, x))
end

forward(rng::AbstractRNG, d::Distribution) = _forward(d, rand(rng, d))
function forward(rng::AbstractRNG, d::Distribution, num_samples::Int)
    return _forward(d, rand(rng, d, num_samples))
end
function _forward(d::Distribution, x)
    y, logjac = forward(IdentityBijector, x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf(d, x))
end

function _forward(td::Transformed, x)
    y, logjac = forward(td.transform, x)
    return (
        x = x,
        y = y,
        logabsdetjac = logjac,
        logpdf = logpdf_forward(td, x, logjac)
    )
end
function forward(rng::AbstractRNG, td::Transformed)
    return _forward(td, rand(rng, td.dist))
end
function forward(rng::AbstractRNG, td::Transformed, num_samples::Int)
    return _forward(td, rand(rng, td.dist, num_samples))
end

"""
    forward(d::Distribution)
    forward(d::Distribution, num_samples::Int)

Returns a `NamedTuple` with fields `x`, `y`, `logabsdetjac` and `logpdf`.

In the case where `d isa TransformedDistribution`, this means
- `x = rand(d.dist)`
- `y = d.transform(x)`
- `logabsdetjac` is the logabsdetjac of the "forward" transform.
- `logpdf` is the logpdf of `y`, not `x`

In the case where `d isa Distribution`, this means
- `x = rand(d)`
- `y = x`
- `logabsdetjac = 0.0`
- `logpdf` is logpdf of `x`
"""
forward(d::Distribution) = forward(GLOBAL_RNG, d)
forward(d::Distribution, num_samples::Int) = forward(GLOBAL_RNG, d, num_samples)
