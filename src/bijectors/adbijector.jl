"""
Abstract type for a `Bijector{N}` making use of auto-differentation (AD) to
implement `jacobian` and, by impliciation, `logabsdetjac`.
"""
abstract type ADBijector{AD, N} <: Bijector{N} end

struct SingularJacobianException{B<:Bijector} <: Exception
    b::B
end
function Base.showerror(io::IO, e::SingularJacobianException)
    print(io, "jacobian of $(e.b) is singular")
end

# concrete implementations with optional dependencies ForwardDiff and Tracker
function jacobian end

# TODO: allow batch-computation, especially for univariate case?
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
function logabsdetjac(b::ADBijector, x::Real)
    res = log(abs(jacobian(b, x)))
    return isfinite(res) ? res : throw(SingularJacobianException(b))
end

function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? logabsdet(fact)[1] : throw(SingularJacobianException(b))
end
