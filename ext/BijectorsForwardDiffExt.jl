module BijectorsForwardDiffExt

if isdefined(Base, :get_extension)
    using ForwardDiff: ForwardDiff
else
    using .ForwardDiff: ForwardDiff
end

using Bijectors
import Bijectors::_eps, find_alpha

_eps(::Type{<:ForwardDiff.Dual{<:Any,Real}}) = _eps(Real)
_eps(::Type{<:ForwardDiff.Dual{<:Any,<:Integer}}) = _eps(Real)

# Define forward-mode rule for ForwardDiff and don't trust support for ForwardDiff in Roots
# https://github.com/JuliaMath/Roots.jl/issues/314
function find_alpha(
    wt_y::ForwardDiff.Dual{T,<:Real},
    wt_u_hat::ForwardDiff.Dual{T,<:Real},
    b::ForwardDiff.Dual{T,<:Real},
) where {T}
    # Compute primal
    value_wt_y = ForwardDiff.value(wt_y)
    value_wt_u_hat = ForwardDiff.value(wt_u_hat)
    value_b = ForwardDiff.value(b)
    Ω = find_alpha(value_wt_y, value_wt_u_hat, value_b)

    # Compute derivative
    partials_wt_y = ForwardDiff.partials(wt_y)
    partials_wt_u_hat = ForwardDiff.partials(wt_u_hat)
    partials_b = ForwardDiff.partials(b)
    x = inv(1 + value_wt_u_hat * sech(Ω + value_b)^2)
    ∂Ω = x * (partials_wt_y - tanh(Ω + value_b) * partials_wt_u_hat) + (x - 1) * partials_b

    return ForwardDiff.Dual{T}(Ω, ∂Ω)
end

end
