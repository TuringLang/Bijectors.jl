# differentation rule for the iterative algorithm in the inverse of `PlanarLayer`
ChainRulesCore.@scalar_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(
        x = inv(1 + wt_u_hat * sech(Ω + b)^2),
    ),
    (x, - tanh(Ω + b) * x, x - 1),
)
