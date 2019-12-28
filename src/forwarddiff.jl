using .Bijectors
import ForwardDiff

using .Bijectors: ForwardDiffAD
import .Bijectors: _eps, jacobian

_eps(::Type{<:ForwardDiff.Dual{<:Any, Real}}) = _eps(Real)

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inversed{<:ADBijector{<:ForwardDiffAD}}},
    x::Real
)
    return ForwardDiff.derivative(b, x)
end
function jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inversed{<:ADBijector{<:ForwardDiffAD}}},
    x::AbstractVector{<:Real}
)
    return ForwardDiff.jacobian(b, x)
end