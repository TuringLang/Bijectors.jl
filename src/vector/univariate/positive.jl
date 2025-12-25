# Bijectors for continuous univariate distributions which have support over the positive (or
# non-negative) real numbers.

"""
    ExpOnly(lower)

Callable struct, defined such that `(e::ExpOnly)(x) = exp(x[]) + lower`.

!!! warning
    This does not check whether the input has exactly one element.
"""
struct ExpOnly{L<:Real}
    lower::L
end
(e::ExpOnly)(y::AbstractVector{<:Real}) = exp(y[]) + e.lower
function with_logabsdet_jacobian(e::ExpOnly, y::AbstractVector{<:Real})
    yi = y[]
    x = exp(yi)
    return (x + e.lower, yi)
end
inverse(e::ExpOnly) = LogVect(e.lower)

"""
   LogVect(lower)

Callable struct, defined such that `(::LogVect)(x) = [log(x - lower)]`.

!!! warning
    This does not check whether the input is a scalar greater than `lower`.
"""
struct LogVect{L<:Real}
    lower::L
end
(l::LogVect)(x::Real) = [log(x - l.lower)]
function with_logabsdet_jacobian(l::LogVect, x::Real)
    logx = log(x - l.lower)
    return ([logx], -logx)
end
inverse(::LogVect) = ExpOnly()

for dist_type in [
    D.BetaPrime,
    D.Chi,
    D.Chisq,
    D.Erlang,
    D.Exponential,
    D.FDist,
    # Wikipedia's definition of the Frechet distribution allows for a location parameter,
    # which could cause its minimum to be nonzero. However, Distributions.jl's `Frechet`
    # does not implement this, so we can lump it in here.
    D.Frechet,
    D.Gamma,
    D.InverseGamma,
    D.InverseGaussian,
    D.Kolmogorov,
    D.Lindley,
    D.LogNormal,
    D.NoncentralChisq,
    D.NoncentralF,
    D.Rayleigh,
    D.Rician,
    D.StudentizedRange,
    D.Weibull,
]
    @eval begin
        VectorBijectors.from_linked_vec(d::$dist_type) = ExpOnly(minimum(d))
        VectorBijectors.to_linked_vec(d::$dist_type) = LogVect(minimum(d))
    end
end
