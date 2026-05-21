"""
    CDF(dist) <: ScalarToScalarBijector

Map a scalar from the support of `dist` to the unit interval using the CDF of `dist`.

This is the scalar-to-scalar counterpart of [`Bijectors.CDFBijector`](@ref), suitable for
use inside the `Bijectors.VectorBijectors` framework. Its inverse is [`Quantile`](@ref).

!!! warning
    This does not check whether the input is in the support of `dist`.
"""
struct CDF{Dist<:D.ContinuousUnivariateDistribution} <: ScalarToScalarBijector
    dist::Dist
end
B.is_monotonically_increasing(::CDF) = true
B.is_monotonically_decreasing(::CDF) = false
(c::CDF)(x::Real) = D.cdf(c.dist, x)
with_logabsdet_jacobian(c::CDF, x::Real) = (D.cdf(c.dist, x), D.logpdf(c.dist, x))

"""
    Quantile(dist) <: ScalarToScalarBijector

Map a scalar from the unit interval to the support of `dist` using the quantile function of
`dist`.

This is the scalar-to-scalar counterpart of [`Bijectors.QuantileBijector`](@ref), suitable
for use inside the `Bijectors.VectorBijectors` framework. Its inverse is [`CDF`](@ref).

!!! warning
    This does not check whether the input is in the unit interval.
"""
struct Quantile{Dist<:D.ContinuousUnivariateDistribution} <: ScalarToScalarBijector
    dist::Dist
end
B.is_monotonically_increasing(::Quantile) = true
B.is_monotonically_decreasing(::Quantile) = false
(q::Quantile)(x::Real) = D.quantile(q.dist, x)
function with_logabsdet_jacobian(q::Quantile, x::Real)
    y = D.quantile(q.dist, x)
    return (y, -D.logpdf(q.dist, y))
end

inverse(c::CDF) = Quantile(c.dist)
inverse(q::Quantile) = CDF(q.dist)
