import Base: ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf

const Elementwise{F} = Base.Fix1{<:Union{typeof(map),typeof(broadcast)},F}
"""
    elementwise(f)

Alias for `Base.Fix1(broadcast, f)`.

In the case where `f::ComposedFunction`, the result is
`Base.Fix1(broadcast, f.outer) ∘ Base.Fix1(broadcast, f.inner)` rather than
`Base.Fix1(broadcast, f)`.
"""
elementwise(f) = Base.Fix1(broadcast, f)
elementwise(f::typeof(identity)) = identity
# TODO: This is makes dispatching quite a bit easier, but uncertain if this is really
# the way to go.
function elementwise(f::ComposedFunction)
    return ComposedFunction(elementwise(f.outer), elementwise(f.inner))
end
const Columnwise{F} = Base.Fix1{typeof(eachcolmaphcat),F}
"""

Alias for `Base.Fix1(eachcolmaphcat, f)`.

Represents a function `f` which is applied to each column of an input.
"""
columnwise(f) = Base.Fix1(eachcolmaphcat, f)
inverse(f::Columnwise) = columnwise(inverse(f.x))

transform(f::Columnwise, x::AbstractMatrix) = f(x)
function logabsdetjac(f::Columnwise, x::AbstractMatrix)
    return sum(Base.Fix1(logabsdetjac, f.x), eachcol(x))
end
with_logabsdet_jacobian(f::Columnwise, x::AbstractMatrix) = (f(x), logabsdetjac(f, x))

"""
    output_size(f, sz)

Returns the output size of `f` given the input size `sz`.
"""
output_size(f, sz) = sz
output_size(f::ComposedFunction, sz) = output_size(f.outer, output_size(f.inner, sz))

"""
    output_length(f, len::Int)

Returns the output length of `f` given the input length `len`.
"""
output_length(f, len::Int) = only(output_size(f, (len,)))

######################
# Bijector interface #
######################
"""

Abstract type for a transformation.

# Implementing

A subtype of `Transform` of should at least implement [`transform(b, x)`](@ref).

If the `Transform` is also invertible:
- Required:
  - _Either_ of the following:
    - `transform(::Inverse{<:MyTransform}, x)`: the `transform` for its inverse.
    - `InverseFunctions.inverse(b::MyTransform)`: returns an existing `Transform`.
  - [`logabsdetjac`](@ref): computes the log-abs-det jacobian factor.
- Optional:
  - `with_logabsdet_jacobian`: `transform` and `logabsdetjac` combined. Useful in cases where we
    can exploit shared computation in the two.

For the above methods, there are mutating versions which can _optionally_ be implemented:
- [`with_logabsdet_jacobian!`](@ref)
- [`logabsdetjac!`](@ref)
- [`with_logabsdet_jacobian!`](@ref)
"""
abstract type Transform end

(t::Transform)(x) = transform(t, x)

Broadcast.broadcastable(b::Transform) = Ref(b)

"""
    transform(b, x)

Transform `x` using `b`, treating `x` as a single input.
"""
transform(f::F, x) where {F<:Function} = f(x)
function transform(t::Transform, x)
    res = with_logabsdet_jacobian(t, x)
    if res isa ChangesOfVariables.NoLogAbsDetJacobian
        error(
            "`transform` not implemented for $(typeof(f)); implement `transform` and/or `with_logabsdet_jacobian`.",
        )
    end

    return first(res)
end

"""
    transform!(b, x[, y])

Transform `x` using `b`, storing the result in `y`.

If `y` is not provided, `x` is used as the output.
"""
transform!(b, x) = transform!(b, x, x)
transform!(b, x, y) = copyto!(y, transform(b, x))

"""
    logabsdetjac(b, x)

Return `log(abs(det(J(b, x))))`, where `J(b, x)` is the jacobian of `b` at `x`.
"""
function logabsdetjac(b, x)
    res = with_logabsdet_jacobian(b, x)
    if res isa ChangesOfVariables.NoLogAbsDetJacobian
        error(
            "`logabsdetjac` not implemented for $(typeof(b)); implement `logabsdetjac` and/or `with_logabsdet_jacobian`.",
        )
    end

    return last(res)
end

"""
    logabsdetjac!(b, x[, logjac])

Compute `log(abs(det(J(b, x))))` and store the result in `logjac`, where `J(b, x)` is the jacobian of `b` at `x`.
"""
logabsdetjac!(b, x) = logabsdetjac!(b, x, zero(eltype(x)))
logabsdetjac!(b, x, logjac) = (logjac += logabsdetjac(b, x))

"""
    with_logabsdet_jacobian!(b, x[, y, logjac])

Compute `transform(b, x)` and `logabsdetjac(b, x)`, storing the result
in `y` and `logjac`, respetively.

If `y` is not provided, then `x` will be used in its place.

Defaults to calling `with_logabsdet_jacobian(b, x)` and updating `y` and `logjac` with the result.
"""
with_logabsdet_jacobian!(b, x) = with_logabsdet_jacobian!(b, x, x)
with_logabsdet_jacobian!(b, x, y) = with_logabsdet_jacobian!(b, x, y, zero(eltype(x)))
function with_logabsdet_jacobian!(b, x, y, logjac)
    y_, logjac_ = with_logabsdet_jacobian(b, x)
    y .= y_
    return (y, logjac + logjac_)
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

"""
    isinvertible(t)

Return `true` if `t` is invertible, and `false` otherwise.
"""
isinvertible(t) = inverse(t) !== InverseFunctions.NoInverse()

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
    inverse(t::Transform)

Returns the inverse of transform `t`.
"""
inverse(t::Transform) = Inverse(t)
inverse(ib::Inverse) = ib.orig

Base.:(==)(b1::Inverse, b2::Inverse) = b1.orig == b2.orig

"Abstract type of a bijector, i.e. differentiable bijection with differentiable inverse."
abstract type Bijector <: Transform end

isinvertible(::Bijector) = true

# Default implementation for inverse of a `Bijector`.
logabsdetjac(ib::Inverse{<:Transform}, y) = -logabsdetjac(ib.orig, transform(ib, y))

function with_logabsdet_jacobian(ib::Inverse{<:Transform}, y)
    x = transform(ib, y)
    return x, -logabsdetjac(inverse(ib), x)
end

"""
    logabsdetjacinv(b, y)

Just an alias for `logabsdetjac(inverse(b), y)`.
"""
logabsdetjacinv(b, y) = logabsdetjac(inverse(b), y)

##############################
# Example bijector: identity #
##############################
transform(::typeof(identity), x) = copy(x)
transform!(::typeof(identity), x, y) = copy!(y, x)

logabsdetjac(::typeof(identity), x) = zero(eltype(x))
logabsdetjac!(::typeof(identity), x, logjac) = logjac

######################
# Bijectors includes #
######################
# General
include("bijectors/composed.jl")
include("bijectors/stacked.jl")
include("bijectors/reshape.jl")

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
include("bijectors/product_bijector.jl")

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
