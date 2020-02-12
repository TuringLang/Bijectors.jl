import Base: inv, ∘

import Random: AbstractRNG
import Distributions: logpdf, rand, rand!, _rand!, _logpdf, params

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
"Abstract type for a bijector."
abstract type AbstractBijector end

"Abstract type of bijectors with fixed dimensionality."
abstract type Bijector{N} <:AbstractBijector end

dimension(b::Bijector{N}) where {N} = N
dimension(b::Type{<:Bijector{N}}) where {N} = N

Broadcast.broadcastable(b::Bijector) = Ref(b)

"""
    isclosedform(b::Bijector)::bool
    isclosedform(b⁻¹::Inversed{<:Bijector})::bool

Returns `true` or `false` depending on whether or not evaluation of `b`
has a closed-form implementation.

Most bijectors have closed-form evaluations, but there are cases where
this is not the case. For example the *inverse* evaluation of `PlanarLayer`
requires an iterative procedure to evaluate and thus is not differentiable.
"""
isclosedform(b::Bijector) = true

"""
    inv(b::Bijector)
    Inversed(b::Bijector)

A `Bijector` representing the inverse transform of `b`.
"""
struct Inversed{B <: Bijector, N} <: Bijector{N}
    orig::B

    Inversed(b::B) where {N, B<:Bijector{N}} = new{B, N}(b)
end


inv(b::Bijector) = Inversed(b)
inv(ib::Inversed{<:Bijector}) = ib.orig

"""
    logabsdetjac(b::Bijector, x)
    logabsdetjac(ib::Inversed{<:Bijector}, y)

Computes the log(abs(det(J(b(x))))) where J is the jacobian of the transform.
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


"""
    logabsdetjacinv(b::Bijector, y)

Just an alias for `logabsdetjac(inv(b), y)`.
"""
logabsdetjacinv(b::Bijector, y) = logabsdetjac(inv(b), y)

##############################
# Example bijector: Identity #
##############################

struct Identity{N} <: Bijector{N} end
(::Identity)(x) = x
inv(b::Identity) = b

logabsdetjac(::Identity, x::Real) = zero(eltype(x))
@generated function logabsdetjac(
    b::Identity{N1},
    x::AbstractArray{T2, N2}
) where {N1, T2, N2}
    if N1 == N2
        return :(zero(eltype(x)))
    elseif N1 + 1 == N2
        return :(zeros(eltype(x), size(x, $N2)))
    else
        return :(throw(MethodError(logabsdetjac, (b, x))))
    end
end

########################
# Convenient constants #
########################
const ZeroOrOneDimBijector = Union{Bijector{0}, Bijector{1}}

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
include("bijectors/truncated.jl")
include("bijectors/distribution_bijector.jl")

# Normalizing flow related
include("bijectors/planar_layer.jl")
include("bijectors/radial_layer.jl")
include("bijectors/normalise.jl")

##################
# Other includes #
##################
include("transformed_distribution.jl")
