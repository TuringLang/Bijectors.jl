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
"Abstract type for a bijector."
abstract type AbstractBijector end

"Abstract type of bijectors with fixed dimensionality."
abstract type Bijector{N} <: AbstractBijector end

dimension(b::Bijector{N}) where {N} = N
dimension(b::Type{<:Bijector{N}}) where {N} = N

Broadcast.broadcastable(b::Bijector) = Ref(b)

"""
    isclosedform(b::Bijector)::bool
    isclosedform(b⁻¹::Inverse{<:Bijector})::bool

Returns `true` or `false` depending on whether or not evaluation of `b`
has a closed-form implementation.

Most bijectors have closed-form evaluations, but there are cases where
this is not the case. For example the *inverse* evaluation of `PlanarLayer`
requires an iterative procedure to evaluate and thus is not differentiable.
"""
isclosedform(b::Bijector) = true

"""
    inv(b::Bijector)
    Inverse(b::Bijector)

A `Bijector` representing the inverse transform of `b`.
"""
struct Inverse{B <: Bijector, N} <: Bijector{N}
    orig::B

    Inverse(b::B) where {N, B<:Bijector{N}} = new{B, N}(b)
end
up1(b::Inverse) = Inverse(up1(b.orig))

inv(b::Bijector) = Inverse(b)
inv(ib::Inverse{<:Bijector}) = ib.orig
Base.:(==)(b1::Inverse{<:Bijector}, b2::Inverse{<:Bijector}) = b1.orig == b2.orig

"""
    logabsdetjac(b::Bijector, x)
    logabsdetjac(ib::Inverse{<:Bijector}, y)

Computes the log(abs(det(J(b(x))))) where J is the jacobian of the transform.
Similarily for the inverse-transform.

Default implementation for `Inverse{<:Bijector}` is implemented as
`- logabsdetjac` of original `Bijector`.
"""
logabsdetjac(ib::Inverse{<:Bijector}, y) = - logabsdetjac(ib.orig, ib(y))

"""
    forward(b::Bijector, x)

Computes both `transform` and `logabsdetjac` in one forward pass, and
returns a named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`.

This defaults to the call above, but often one can re-use computation
in the computation of the forward pass and the computation of the
`logabsdetjac`. `forward` allows the user to take advantange of such
efficiencies, if they exist.
"""
forward(b::Bijector, x) = (rv=b(x), logabsdetjac=logabsdetjac(b, x))

"""
    logabsdetjacinv(b::Bijector, y)

Just an alias for `logabsdetjac(inv(b), y)`.
"""
logabsdetjacinv(b::Bijector, y) = logabsdetjac(inv(b), y)

##############################
# Example bijector: Identity #
##############################

struct Identity{N} <: Bijector{N} end
(::Identity)(x) = copy(x)
inv(b::Identity) = b
up1(::Identity{N}) where {N} = Identity{N + 1}()

logabsdetjac(::Identity{0}, x::Real) = zero(eltype(x))
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
logabsdetjac(::Identity{2}, x::AbstractArray{<:AbstractMatrix}) = zeros(eltype(x[1]), size(x))

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
include("bijectors/pd.jl")
include("bijectors/corr.jl")
include("bijectors/truncated.jl")

# Normalizing flow related
include("bijectors/planar_layer.jl")
include("bijectors/radial_layer.jl")
include("bijectors/leaky_relu.jl")
include("bijectors/coupling.jl")
include("bijectors/normalise.jl")

##################
# Other includes #
##################
include("transformed_distribution.jl")
