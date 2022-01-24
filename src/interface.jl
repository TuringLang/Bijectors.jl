import Base: ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf

const Elementwise{F} = Base.Fix1{<:Union{typeof(map),typeof(broadcast)}, F}
"""
    elementwise(f)

Alias for `Base.Fix1(broadcast, f)`.

In the case where `f::ComposedFunction`, the result is
`Base.Fix1(broadcast, f.outer) ∘ Base.Fix1(broadcast, f.inner)` rather than
`Base.Fix1(broadcast, f)`.
"""
elementwise(f) = Base.Fix1(broadcast, f)
# TODO: This is makes dispatching quite a bit easier, but uncertain if this is really
# the way to go.
elementwise(f::ComposedFunction) = ComposedFunction(elementwise(f.outer), elementwise(f.inner))

#######################################
# AD stuff "extracted" from Turing.jl #
#######################################

abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct ReverseDiffAD <: ADBackend end
struct TrackerAD <: ADBackend end
struct ZygoteAD <: ADBackend end

const ADBACKEND = Ref(:forwarddiff)
setadbackend(backend_sym::Symbol) = setadbackend(Val(backend_sym))
setadbackend(::Val{:forwarddiff}) = ADBACKEND[] = :forwarddiff
setadbackend(::Val{:reversediff}) = ADBACKEND[] = :reversediff
setadbackend(::Val{:tracker}) = ADBACKEND[] = :tracker
setadbackend(::Val{:zygote}) = ADBACKEND[] = :zygote

ADBackend() = ADBackend(ADBACKEND[])
ADBackend(T::Symbol) = ADBackend(Val(T))
ADBackend(::Val{:forwarddiff}) = ForwardDiffAD
ADBackend(::Val{:reversediff}) = ReverseDiffAD
ADBackend(::Val{:tracker}) = TrackerAD
ADBackend(::Val{:zygote}) = ZygoteAD
ADBackend(::Val) = error("The requested AD backend is not available. Make sure to load all required packages.")

######################
# Bijector interface #
######################
"""

Abstract type for a transformation.

# Implementing

A subtype of `Transform` of should at least implement [`transform(b, x)`](@ref).

If the `Transform` is also invertible:
- Required:
  - [`invertible`](@ref): should return [`Invertible`](@ref).
  - _Either_ of the following:
    - `transform(::Inverse{<:MyTransform}, x)`: the `transform` for its inverse.
    - `InverseFunctions.inverse(b::MyTransform)`: returns an existing `Transform`.
  - [`logabsdetjac`](@ref): computes the log-abs-det jacobian factor.
- Optional:
  - [`forward`](@ref): `transform` and `logabsdetjac` combined. Useful in cases where we
    can exploit shared computation in the two.

For the above methods, there are mutating versions which can _optionally_ be implemented:
- [`transform_single!`](@ref)
- [`logabsdetjac_single!`](@ref)
- [`forward_single!`](@ref)

Finally, there are _batched_ versions of the above methods which can _optionally_ be implemented:
- [`transform_multiple`](@ref)
- [`logabsdetjac_multiple`](@ref)
- [`forward_multiple`](@ref)

and similarly for the mutating versions. Default implementations depends on the type of `xs`.
"""
abstract type Transform end

(t::Transform)(x) = transform(t, x)

Broadcast.broadcastable(b::Transform) = Ref(b)

"""
    transform(b, x)

Transform `x` using `b`, treating `x` as a single input.

Alias for [`transform_single(b, x)`](@ref).
"""
transform(b, x) = transform_single(b, x)

"""
    transform_single(b, x)

Transform `x` using `b`.

Defaults to `b(x)` if `b isa Function` and `first(forward(b, x))` otherwise.
"""
transform_single(b, x) = first(forward(b, x))
transform_single(b::Function, x) = b(x)

"""
    transform(b, xs::AbstractBatch)

Transform `xs` using `b`, treating `xs` as a collection of inputs.

Alias for [`transform_multiple(b, x)`](@ref).
"""
transform(b, xs::AbstractBatch) = transform_multiple(b, xs)

"""
    transform_multiple(b, xs)

Transform `xs` using `b`, treating `xs` as a collection of inputs.

Defaults to `map(Base.Fix1(transform, b), xs)`.
"""
transform_multiple(b, xs) = map(Base.Fix1(transform, b), xs)
function transform_multiple(b::Elementwise, x::Batching.ArrayBatch)
    return batch_like(x, transform(b, Batching.value(x)))
end


"""
    transform!(b, x[, y])

Transform `x` using `b`, storing the result in `y`.

If `y` is not provided, `x` is used as the output.

Alias for [`transform_single!(b, x, y)`](@ref).
"""
transform!(b, x, y=x) = transform_single!(b, x, y)

"""
    transform_single!(b, x, y)

Transform `x` using `b`, storing the result in `y`.
"""
transform_single!(b, x, y) = (y .= transform(b, x))

"""
    transform!(b, xs::AbstractBatch[, ys::AbstractBatch])

Transform `x` for `x` in `xs` using `b`, storing the result in `ys`.

If `ys` is not provided, `xs` is used as the output.

Alias for [`transform_multiple!(b, xs, ys)`](@ref).
"""
transform!(b, xs::AbstractBatch, ys::AbstractBatch=xs) = transform_multiple!(b, xs, ys)

"""
    transform_multiple!(b, xs::AbstractBatch[, ys::AbstractBatch])

Transform `x` for `x` in `xs` using `b`, storing the result in `ys`.
"""
transform_multiple!(b, xs, ys) = broadcast!(Base.Fix1(transform, b), xs, ys)
function transform_multiple!(b::Elementwise, x::Batching.ArrayBatch, y::Batching.ArrayBatch)
    broadcast!(b, Batching.value(y), Batching.value(x))
    return y
end

"""
    logabsdetjac(b, x)

Return `log(abs(det(J(b, x))))`, where `J(b, x)` is the jacobian of `b` at `x`.

Alias for [`logabsdetjac_single`](@ref).

See also: [`logabsdetjac(b, xs::AbstractBatch)`](@ref).
"""
logabsdetjac(b, x) = logabsdetjac_single(b, x)

"""
    logabsdetjac_single(b, x)

Return `log(abs(det(J(b, x))))`, where `J(b, x)` is the jacobian of `b` at `x`.

Defaults to `last(forward(b, x))`.
"""
logabsdetjac_single(b, x) = last(forward(b, x))

"""
    logabsdetjac(b, xs::AbstractBatch)

Return a collection representing `log(abs(det(J(b, x))))` for every `x` in `xs`, 
where `J(b, x)` is the jacobian of `b` at `x`.

Alias for [`Bijectors.logabsdetjac_multiple`](@ref).

See also: [`logabsdetjac(b, x)`](@ref).
"""
logabsdetjac(b, xs::AbstractBatch) = logabsdetjac_multiple(b, xs)

"""
    logabsdetjac_multiple(b, xs)

Return a collection representing `log(abs(det(J(b, x))))` for every `x` in `xs`, 
where `J(b, x)` is the jacobian of `b` at `x`.

Defaults to `map(Base.Fix1(logabsdetjac, b), xs)`.
"""
logabsdetjac_multiple(b, xs) = map(Base.Fix1(logabsdetjac, b), xs)

"""
    logabsdetjac!(b, x[, logjac])

Compute `log(abs(det(J(b, x))))` and store the result in `logjac`, where `J(b, x)` is the jacobian of `b` at `x`.

Alias for [`logabsdetjac_single!(b, x, logjac)`](@ref).
"""
logabsdetjac!(b, x, logjac=zero(eltype(x))) = logabsdetjac_single!(b, x, logjac)

"""
    logabsdetjac_single!(b, x[, logjac])

Compute `log(abs(det(J(b, x))))` and accumulate the result in `logjac`, 
where `J(b, x)` is the jacobian of `b` at `x`.
"""
logabsdetjac_single!(b, x, logjac) = (logjac += logabsdetjac(b, x))

"""
    logabsdetjac!(b, xs::AbstractBatch[, logjacs::AbstractBatch])

Compute `log(abs(det(J(b, x))))` and store the result in `logjacs` for
every `x` in `xs`, where `J(b, x)` is the jacobian of `b` at `x`.

Alias for [`logabsdetjac_single!(b, x, logjac)`](@ref).
"""
function logabsdetjac!(
    b,
    xs::AbstractBatch,
    logjacs::AbstractBatch=batch_like(xs, zeros(eltype(eltype(xs)), length(xs)))
)
    return logabsdetjac_multiple!(b, xs, logjacs)
end

"""
    logabsdetjac_multiple!(b, xs::AbstractBatch, logjacs::AbstractBatch)

Compute `log(abs(det(J(b, x))))` and store the result in `logjacs` for
every `x` in `xs`, where `J(b, x)` is the jacobian of `b` at `x`.
"""
logabsdetjac_multiple!(b, xs, logjacs) = (logjacs .+= logabsdetjac(b, xs))

"""
    forward(b, x)

Return `(transform(b, x), logabsdetjac(b, x))` treating `x` as single input.

Alias for [`forward_single(b, x)`](@ref).

See also: [`forward(b, xs::AbstractBatch)`](@ref).
"""
forward(b, x) = forward_single(b, x)

"""
    forward_single(b, x)

Return `(transform(b, x), logabsdetjac(b, x))` treating `x` as a single input.

Defaults to `ChangeOfVariables.with_logabsdet_jacobian(b, x)`.
"""
forward_single(b, x) = with_logabsdet_jacobian(b, x)

"""
    forward(b, xs::AbstractBatch)

Return `(transform(b, x), logabsdetjac(b, x))` treating `x` as
collection of inputs.

Alias for [`forward_multiple(b, xs)`](@ref).

See also: [`forward(b, x)`](@ref).
"""
forward(b, xs::AbstractBatch) = forward_multiple(b, xs)

"""
    forward_multiple(b, xs)

Return `(transform(b, xs), logabsdetjac(b, xs))` treating
`xs` as a batch, i.e. a collection inputs.

See also: [`forward_single(b, x)`](@ref).
"""
function forward_multiple(b, xs)
    # If `b` doesn't have its own definition of `forward_multiple`
    # we just broadcast `forward_single`, resulting in a batch of `(y, logjac)`
    # pairs which we then unwrap.
    results = forward.(Ref(b), xs)
    ys = map(first, results)
    logjacs = map(last, results)
    return batch_like(xs, ys, logjacs)
end

# `logjac` as an argument doesn't make too much sense for `forward_single!`
# when the inputs have `eltype` `Real`.
"""
    forward!(b, x[, y, logjac])

Compute `transform(b, x)` and `logabsdetjac(b, x)`, storing the result
in `y` and `logjac`, respetively.

If `y` is not provided, then `x` will be used in its place.

Alias for [`forward_single!(b, x, y, logjac)`](@ref).

See also: [`forward!(b, xs::AbstractBatch, ys::AbstractBatch, logjacs::AbstractBatch)`](@ref).
"""
forward!(b, x, y=x, logjac=zero(eltype(x))) = forward_single!(b, x, y, logjac)

"""
    forward_single!(b, x, y, logjac)

Compute `transform(b, x)` and `logabsdetjac(b, x)`, storing the result 
in `y` and `logjac`, respetively.

Defaults to calling `forward(b, x)` and updating `y` and `logjac` with the result.
"""
function forward_single!(b, x, y, logjac)
    y_, logjac_ = forward(b, x)
    y .= y_
    return (y, logjac + logjac_)
end

"""
    forward!(b, xs[, ys, logjacs])

Compute `transform(b, x)` and `logabsdetjac(b, x)` for every `x` in the collection `xs`, 
storing the results in `ys` and `logjacs`, respetively.

If `ys` is not provided, then `xs` will be used in its place.

Alias for [`forward_multiple!(b, xs, ys, logjacs)`](@ref).

See also: [`forward!(b, x, y, logjac)`](@ref).
"""
function forward!(
    b,
    xs::AbstractBatch,
    ys::AbstractBatch=xs,
    logjacs::AbstractBatch=batch_like(xs, zeros(eltype(eltype(xs)), length(xs)))
)
    return forward_multiple!(b, xs, ys, logjacs)
end

"""
    forward!(b, xs, ys, logjacs)

Compute `transform(b, x)` and `logabsdetjac(b, x)` for every `x` in the collection `xs`, 
storing the results in `ys` and `logjacs`, respetively.

Defaults to iterating through the `xs` and calling [`forward_single(b, x)`](@ref)
for every `x` in `xs`.
"""
function forward_multiple!(b, xs, ys, logjacs)
    for i in eachindex(ys)
        res = forward_single(b, xs[i])
        ys[i] .= first(res)
        logjacs[i] += last(res)
    end

    return ys, logjacs
end

"""
    isclosedform(b::Transform)::bool
    isclosedform(b⁻¹::Inverse{<:Transform})::bool

Returns `true` or `false` depending on whether or not evaluation of `b`
has a closed-form implementation.

Most transformations have closed-form evaluations, but there are cases where
this is not the case. For example the *inverse* evaluation of `PlanarLayer`
requires an iterative procedure to evaluate.
"""
isclosedform(t::Transform) = true

# Invertibility "trait".
struct NotInvertible end
struct Invertible end

# Useful for checking if compositions, etc. are invertible or not.
Base.:+(::NotInvertible, ::Invertible) = NotInvertible()
Base.:+(::Invertible, ::NotInvertible) = NotInvertible()
Base.:+(::NotInvertible, ::NotInvertible) = NotInvertible()
Base.:+(::Invertible, ::Invertible) = Invertible()

"""
    invertible(t)

Return `Invertible()` if `t` is invertible, and `NotInvertible()` otherwise.
"""
invertible(::Transform) = NotInvertible()

"""
    isinvertible(t)

Return `true` if `t` is invertible, and `false` otherwise.
"""
isinvertible(t::Transform) = invertible(t) isa Invertible

"""
    inverse(b::Transform)
    Inverse(b::Transform)

A `Transform` representing the inverse transform of `b`.
"""
struct Inverse{T<:Transform} <: Transform
    orig::T

    function Inverse(orig::Transform)
        if !isinvertible(orig)
            error("$(orig) is not invertible")
        end

        return new{typeof(orig)}(orig)
    end
end

Functors.@functor Inverse

"""
    inverse(t::Transform[, ::Invertible])

Returns the inverse of transform `t`.
"""
inverse(t::Transform) = Inverse(t)
inverse(ib::Inverse) = ib.orig

invertible(ib::Inverse) = Invertible()

Base.:(==)(b1::Inverse, b2::Inverse) = b1.orig == b2.orig

"Abstract type of a bijector, i.e. differentiable bijection with differentiable inverse."
abstract type Bijector <: Transform end

invertible(::Bijector) = Invertible()

# Default implementation for inverse of a `Bijector`.
logabsdetjac(ib::Inverse{<:Bijector}, y) = -logabsdetjac(ib.orig, ib(y))

"""
    logabsdetjacinv(b::Bijector, y)

Just an alias for `logabsdetjac(inverse(b), y)`.
"""
logabsdetjacinv(b::Bijector, y) = logabsdetjac(inverse(b), y)

##############################
# Example bijector: Identity #
##############################
# Here we don't need to separate between batched version and non-batched, and so
# we can just overload `transform`, etc. directly.
transform(::typeof(identity), x) = copy(x)
transform!(::typeof(identity), x, y) = copy!(y, x)

logabsdetjac_single(::typeof(identity), x) = zero(eltype(x))
logabsdetjac_multiple(::typeof(identity), x) = batch_like(x, zeros(eltype(eltype(x)), length(x)))

logabsdetjac_single!(::typeof(identity), x, logjac) = logjac
logabsdetjac_multiple!(::typeof(identity), x, logjac) = logjac

####################
# Batched versions #
####################
# NOTE: This needs to be after we've defined some `transform`, `logabsdetjac`, etc.
# so we can actually reference them. Since we just did this for `Identity`, we're good.
# Broadcast.broadcasted(b::Transform, xs::Batch) = transform_multiple(b, xs)
# Broadcast.broadcasted(::typeof(transform), b::Transform, xs::Batch) = transform_multiple(b, xs)
# Broadcast.broadcasted(::typeof(logabsdetjac), b::Transform, xs::Batch) = logabsdetjac_multiple(b, xs)
# Broadcast.broadcasted(::typeof(forward), b::Transform, xs::Batch) = forward_multiple(b, xs)

######################
# Bijectors includes #
######################
# General
# include("bijectors/adbijector.jl")
include("bijectors/composed.jl")
# include("bijectors/stacked.jl")

# Specific
include("bijectors/exp_log.jl")
include("bijectors/logit.jl")
# include("bijectors/scale.jl")
# include("bijectors/shift.jl")
# include("bijectors/permute.jl")
# include("bijectors/simplex.jl")
# include("bijectors/pd.jl")
# include("bijectors/corr.jl")
# include("bijectors/truncated.jl")
# include("bijectors/named_bijector.jl")
# include("bijectors/ordered.jl")

# Normalizing flow related
# include("bijectors/planar_layer.jl")
# include("bijectors/radial_layer.jl")
# include("bijectors/leaky_relu.jl")
# include("bijectors/coupling.jl")
# include("bijectors/normalise.jl")
# include("bijectors/rational_quadratic_spline.jl")

##################
# Other includes #
##################
include("transformed_distribution.jl")
