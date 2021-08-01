import Base: inv, ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf

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

## Implementing

A subtype of `Transform` of should at least implement `transform(b, x)`.

If the `Transform` is also invertible:
- Required:
  - [`invertible`](@ref): should return [`Invertible`](@ref).
  - _Either_ of the following:
    - `transform(::Inverse{<:MyTransform}, x)`: the `transform` for its inverse.
    - `Base.inv(b::MyTransform)`: returns an existing `Transform`.
  - [`logabsdetjac`](@ref): computes the log-abs-det jacobian factor.
- Optional:
  - [`forward`](@ref): `transform` and `logabsdetjac` combined. Useful in cases where we
    can exploit shared computation in the two.

For the above methods, there are mutating versions which can _optionally_ be implemented:
- [`transform!`](@ref)
- [`logabsdetjac!`](@ref)
- [`forward!`](@ref)

Finally, there are _batched_ versions of the above methods which can _optionally_ be implemented:
- [`transform_batch`](@ref)
- [`logabsdetjac_batch`](@ref)
- [`forward_batch`](@ref)

and similarly for the mutating versions. Default implementations depends on the type of `xs`.
Note that these methods are usually used through broadcasting, i.e. `b.(x)` with `x` a `AbstractBatch`
falls back to `transform_batch(b, x)`.
"""
abstract type Transform end

Broadcast.broadcastable(b::Transform) = Ref(b)

"""
    transform(b, x)

Transform `x` using `b`.

Alternatively, one can just call `b`, i.e. `b(x)`.
"""
transform
(t::Transform)(x) = transform(t, x)

"""
    transform!(b, x, y)

Transforms `x` using `b`, storing the result in `y`.
"""
transform!(b, x, y) = (y .= transform(b, x))

"""
    logabsdetjac(b, x)

Computes the log(abs(det(J(b(x))))) where J is the jacobian of the transform.
"""
logabsdetjac

"""
    logabsdetjac!(b, x, logjac)

Computes the log(abs(det(J(b(x))))) where J is the jacobian of the transform,
_accumulating_ the result in `logjac`.
"""
logabsdetjac!(b, x, logjac) = (logjac += logabsdetjac(b, x))

"""
    forward(b, x)

Computes both `transform` and `logabsdetjac` in one forward pass, and
returns a named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`.

This defaults to the call above, but often one can re-use computation
in the computation of the forward pass and the computation of the
`logabsdetjac`. `forward` allows the user to take advantange of such
efficiencies, if they exist.
"""
forward(b, x) = (result = transform(b, x), logabsdetjac = logabsdetjac(b, x))

function forward!(b, x, out)
    y, logjac = forward(b, x)
    out.result .= y
    out.logabsdetjac .+= logjac

    return out
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
isclosedform(b::Transform) = true

# Invertibility "trait".
struct NotInvertible end
struct Invertible end

# Useful for checking if compositions, etc. are invertible or not.
Base.:+(::NotInvertible, ::Invertible) = NotInvertible()
Base.:+(::Invertible, ::NotInvertible) = NotInvertible()
Base.:+(::NotInvertible, ::NotInvertible) = NotInvertible()
Base.:+(::Invertible, ::Invertible) = Invertible()

invertible(::Transform) = NotInvertible()
isinvertible(t::Transform) = invertible(t) isa Invertible

"""
    inv(b::Transform)
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
    inv(t::Transform[, ::Invertible])

Returns the inverse of transform `t`.
"""
Base.inv(t::Transform) = Inverse(t)
Base.inv(ib::Inverse) = ib.orig

invertible(ib::Inverse) = Invertible()

Base.:(==)(b1::Inverse, b2::Inverse) = b1.orig == b2.orig

"Abstract type of a bijector, i.e. differentiable bijection with differentiable inverse."
abstract type Bijector <: Transform end

invertible(::Bijector) = Invertible()

# Default implementation for inverse of a `Bijector`.
logabsdetjac(ib::Inverse{<:Bijector}, y) = -logabsdetjac(ib.orig, ib(y))

"""
    logabsdetjacinv(b::Bijector, y)

Just an alias for `logabsdetjac(inv(b), y)`.
"""
logabsdetjacinv(b::Bijector, y) = logabsdetjac(inv(b), y)

##############################
# Example bijector: Identity #
##############################

struct Identity <: Bijector end
inv(b::Identity) = b

transform(::Identity, x) = copy(x)
transform!(::Identity, x, y) = (y .= x; return y)
logabsdetjac(::Identity, x) = zero(eltype(x))
logabsdetjac!(::Identity, x, logjac) = logjac

####################
# Batched versions #
####################
# NOTE: This needs to be after we've defined some `transform`, `logabsdetjac`, etc.
# so we can actually reference them. Since we just did this for `Identity`, we're good.
Broadcast.broadcasted(b::Transform, xs::Batch) = transform_batch(b, xs)
Broadcast.broadcasted(::typeof(transform), b::Transform, xs::Batch) = transform_batch(b, xs)
Broadcast.broadcasted(::typeof(logabsdetjac), b::Transform, xs::Batch) = logabsdetjac_batch(b, xs)
Broadcast.broadcasted(::typeof(forward), b::Transform, xs::Batch) = forward_batch(b, xs)


"""
    transform_batch(b, xs)

Transform `xs` by `b`, treating `xs` as a "batch", i.e. a collection of independent inputs.

See also: [`transform`](@ref)
"""
transform_batch(b, xs) = _transform_batch(b, xs)
# Default implementations uses private methods to avoid method ambiguity.
_transform_batch(b, xs::VectorBatch) = reconstruct(xs, map(b, value(xs)))
function _transform_batch(b, xs::ArrayBatch{2})
    # TODO: Check if we can avoid using these custom methods.
    return Batch(eachcolmaphcat(b, value(xs)))
end
function _transform_batch(b, xs::ArrayBatch{N}) where {N}
    res = reduce(map(b, eachslice(value(xs), Val{N}()))) do acc, x
        cat(acc, x; dims = N)
    end
    return reconstruct(xs, res)
end

"""
    transform_batch!(b, xs, ys)

Transform `xs` by `b` treating `xs` as a "batch", i.e. a collection of independent inputs,
and storing the result in `ys`.

See also: [`transform!`](@ref)
"""
transform_batch!(b, xs, ys) = _transform_batch!(b, xs, ys)
function _transform_batch!(b, xs, ys)
    for i = 1:length(xs)
        if eltype(ys) <: Real
            ys[i] = transform(b, xs[i])
        else
            transform!(b, xs[i], ys[i])
        end
    end

    return ys
end

"""
    logabsdetjac_batch(b, xs)

Computes `logabsdetjac(b, xs)`, treating `xs` as a "batch", i.e. a collection of independent inputs.

See also: [`logabsdetjac`](@ref)
"""
logabsdetjac_batch(b, xs) = _logabsdetjac_batch(b, xs)
# Default implementations uses private methods to avoid method ambiguity.
_logabsdetjac_batch(b, xs::VectorBatch) = reconstruct(xs, map(x -> logabsdetjac(b, x), value(xs)))
function _logabsdetjac_batch(b, xs::ArrayBatch{2})
    return reconstruct(xs, map(x -> logabsdetjac(b, x), eachcol(value(xs))))
end
function _logabsdetjac_batch(b, xs::ArrayBatch{N}) where {N}
    return reconstruct(xs, map(x -> logabsdetjac(b, x), eachslice(value(xs), Val{N}())))
end

"""
    logabsdetjac_batch!(b, xs, logjacs)

Computes `logabsdetjac(b, xs)`, treating `xs` as a "batch", i.e. a collection of independent inputs,
accumulating the result in `logjacs`.

See also: [`logabsdetjac!`](@ref)
"""
logabsdetjac_batch!(b, xs, logjacs) = _logabsdetjac_batch!(b, xs, logjacs)
function _logabsdetjac_batch!(b, xs, logjacs)
    for i = 1:length(xs)
        if eltype(logjacs) <: Real
            logjacs[i] += logabsdetjac(b, xs[i])
        else
            logabsdetjac!(b, xs[i], logjacs[i])
        end
    end

    return logjacs
end

"""
    forward_batch(b, xs)

Computes `forward(b, xs)`, treating `xs` as a "batch", i.e. a collection of independent inputs.

See also: [`transform`](@ref)
"""
forward_batch(b, xs) = (result = transform_batch(b, xs), logabsdetjac = logabsdetjac_batch(b, xs))

"""
    forward_batch!(b, xs, out)

Computes `forward(b, xs)` in place, treating `xs` as a "batch", i.e. a collection of independent inputs.

See also: [`forward!`](@ref)
"""
function forward_batch!(b, xs, out)
    transform_batch!(b, xs, out.result)
    logabsdetjac_batch!(b, xs, out.logabsdetjac)

    return out
end

######################
# Bijectors includes #
######################
# General
include("bijectors/adbijector.jl")
include("bijectors/composed.jl")
include("bijectors/stacked.jl")

# Specific
include("bijectors/exp_log.jl")
include("bijectors/logit.jl")
include("bijectors/scale.jl")
include("bijectors/shift.jl")
include("bijectors/permute.jl")
include("bijectors/simplex.jl")
include("bijectors/pd.jl")
include("bijectors/corr.jl")
include("bijectors/truncated.jl")
include("bijectors/named_bijector.jl")
include("bijectors/ordered.jl")

# Normalizing flow related
include("bijectors/planar_layer.jl")
include("bijectors/radial_layer.jl")
include("bijectors/leaky_relu.jl")
include("bijectors/coupling.jl")
include("bijectors/normalise.jl")
include("bijectors/rational_quadratic_spline.jl")

##################
# Other includes #
##################
include("transformed_distribution.jl")
