module BijectorsEnzymeCoreExt

using EnzymeCore

using Bijectors: find_alpha

EnzymeCore.EnzymeRules.@easy_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(x = inv(1 + wt_u_hat * sech(Ω + b)^2),),
    (x, -tanh(Ω + b) * x, x - 1),
)

end  # module
