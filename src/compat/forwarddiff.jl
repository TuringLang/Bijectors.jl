import .ForwardDiff

_eps(::Type{<:ForwardDiff.Dual{<:Any, Real}}) = _eps(Real)
_eps(::Type{<:ForwardDiff.Dual{<:Any, <:Integer}}) = _eps(Real)

# AD implementations
function jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inverse{<:ADBijector{<:ForwardDiffAD}}},
    x::Real
)
    return ForwardDiff.derivative(b, x)
end
function jacobian(
    b::Union{<:ADBijector{<:ForwardDiffAD}, Inverse{<:ADBijector{<:ForwardDiffAD}}},
    x::AbstractVector{<:Real}
)
    return ForwardDiff.jacobian(b, x)
end