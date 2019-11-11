"""
Abstract type for a `Bijector{N}` making use of auto-differentation (AD) to
implement `jacobian` and, by impliciation, `logabsdetjac`.
"""
abstract type ADBijector{AD, N} <: Bijector{N} end

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
    return Tracker.data(Tracker.gradient(b, x)[1])
end
function jacobian(b::Inversed{<:ADBijector{<:TrackerAD}}, y::Real)
    return Tracker.data(Tracker.gradient(b, y)[1])
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
function Base.showerror(io::IO, e::SingularJacobianException)
    print(io, "jacobian of $(e.b) is singular")
end

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
