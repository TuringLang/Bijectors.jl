# differentation rule for the iterative algorithm in the inverse of `PlanarLayer`
ChainRulesCore.@scalar_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(
        x = inv(1 + wt_u_hat * sech(Ω + b)^2),
    ),
    (x, - tanh(Ω + b) * x, x - 1),
)

function ChainRulesCore.rrule(::typeof(_logabsdetjac_shift), a, x)
    return _logabsdetjac_shift(a, x), Δ -> (ChainRulesCore.NO_FIELDS, ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent())
end

function ChainRulesCore.rrule(::typeof(_logabsdetjac_shift_array_batch), a, x)
    return _logabsdetjac_shift_array_batch(a, x), Δ -> (ChainRulesCore.NO_FIELDS, ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent())
end
