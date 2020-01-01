module ForwardDiffCompat

using ..Bijectors: ADBijector, ForwardDiffAD, Inversed
import ..Bijectors: _eps, _jacobian

using ..ForwardDiff: Dual, derivative, jacobian

_eps(::Type{<:Dual{<:Any, Real}}) = _eps(Real)

# AD implementations
function _jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inversed{<:ADBijector{<:ForwardDiffAD}}},
    x::Real
)
    return derivative(b, x)
end
function _jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inversed{<:ADBijector{<:ForwardDiffAD}}},
    x::AbstractVector{<:Real}
)
    return jacobian(b, x)
end

end