# Bijectors for continuous univariate distributions which have support over the positive (or
# non-negative) real numbers.

"""
    Exp(bound, sign) <: ScalarToScalarBijector

Callable struct, defined such that `(e::Exp)(y) = ((e.sign * exp(y)) + e.bound)`. The sign
is determined by the `sign` field.
"""
struct Exp{L<:Real} <: ScalarToScalarBijector
    bound::L
    sign::Int
end
B.is_monotonically_increasing(e::Exp) = e.sign > 0
B.is_monotonically_decreasing(e::Exp) = e.sign < 0
(e::Exp)(y::Real) = first(with_logabsdet_jacobian(e, y))
function with_logabsdet_jacobian(e::Exp, y::Real)
    x = exp(y)
    return ((e.sign * x) + e.bound, y)
end
inverse(e::Exp) = Log(e.bound, e.sign)

"""
   Log(bound, sign) <: ScalarToScalarBijector

Callable struct, defined such that `(l::Log)(x) = log(l.sign * (x - l.bound))`. The sign is
determined by the `sign` field.

This is the appropriate scalar-to-scalar bijector for univariate distributions which have a
support of `[l.bound, ∞)` (if `l.sign == 1`), or `(-∞, l.bound]` (if `l.sign == -1`).

!!! warning
    This does not check whether the input is the domain of the transformation.
"""
struct Log{L<:Real} <: ScalarToScalarBijector
    bound::L
    sign::Int
end
B.is_monotonically_increasing(l::Log) = l.sign > 0
B.is_monotonically_decreasing(l::Log) = l.sign < 0
(l::Log)(x::Real) = first(with_logabsdet_jacobian(l, x))
function with_logabsdet_jacobian(l::Log, x::Real)
    logx = log(l.sign * (x - l.bound))
    return (logx, -logx)
end
inverse(l::Log) = Exp(l.bound, l.sign)

const POSITIVE_UNIVARIATES = Union{
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
}
VectorBijectors.scalar_to_scalar_bijector(d::POSITIVE_UNIVARIATES) = Log(minimum(d), 1)

function VectorBijectors.scalar_to_scalar_bijector(
    d::D.AffineDistribution{<:Any,<:Any,<:POSITIVE_UNIVARIATES}
)
    s = sign(D.scale(d))
    return Log(s > 0 ? minimum(d) : maximum(d), s)
end
