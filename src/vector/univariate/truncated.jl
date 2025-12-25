# This file provides bijectors for continuous univariate distributions which have support
# over arbitrary ranges `(a, b)`. In general, this file has to handle possibly infinite
# values. The reason for this is because of type stability: we can't, for example, return to
# a different bijector type in case we find that the bounds are infinite (which can only be
# determined at runtime). So there is some element of code repetition here: the case where
# both bounds are infinite is the same as `Only` and `Vect`, and the case where only the
# upper bound is infinite is the same as `ExpOnly` and `LogVect`.

using LogExpFunctions: logit, logistic, log1pexp

"""
    TruncateOnly(a, b)

Callable struct, defined such that `(::TruncateOnly(a, b))(x)` converts `x[1]` from
`(-Inf, Inf)` to `(a, b)`.

!!! warning
    This does not check whether the input has exactly one element.
"""
struct TruncateOnly{L<:Real,U<:Real}
    lower::L
    upper::U
end
function (t::TruncateOnly)(y::AbstractVector{<:Real})
    lbounded, ubounded = isfinite(t.lower), isfinite(t.upper)
    return if lbounded && ubounded
        ((t.upper - t.lower) * logistic(y[])) + t.lower
    elseif lbounded
        exp(y[]) + t.lower
    elseif ubounded
        t.upper - exp(y[])
    else
        y[]
    end
end
function with_logabsdet_jacobian(t::TruncateOnly, y::AbstractVector{T}) where {T<:Real}
    lbounded, ubounded = isfinite(t.lower), isfinite(t.upper)
    return if lbounded && ubounded
        bma = t.upper - t.lower
        yi = y[]
        res = (bma * logistic(yi)) + t.lower
        # TODO: Bijectors uses this:
        #    absy = abs(yi)
        #    return log(bma) - absy - (2 * log1pexp(-absy))
        # Check if it's more numerically stable. Don't immediately see a reason why, but I
        # assume there's a reason for it.
        logjac = log(bma) + yi - (2 * log1pexp(yi))
        res, logjac
    elseif lbounded
        yi = y[]
        exp(yi) + t.lower, yi
    elseif ubounded
        yi = y[]
        t.upper - exp(yi), yi
    else
        y[], zero(T)
    end
end
inverse(t::TruncateOnly) = UntruncateVect(t.lower, t.upper)

"""
   UntruncateVect(a, b)

Callable struct, defined such that `(::UntruncateVect(a, b))(x)` converts `x` from `(a, b)`
to a singleton vector whose element lies in `(-Inf, Inf)`.

!!! warning
    This does not check whether the input is a scalar in `(a, b)`.
"""
struct UntruncateVect{L<:Real,U<:Real}
    lower::L
    upper::U
end
function (u::UntruncateVect)(x::Real)
    lbounded, ubounded = isfinite(u.lower), isfinite(u.upper)
    return [
        if lbounded && ubounded
            logit((x - u.lower) / (u.upper - u.lower))
        elseif lbounded
            log(x - u.lower)
        elseif ubounded
            log(u.upper - x)
        else
            x
        end,
    ]
end
function with_logabsdet_jacobian(u::UntruncateVect, x::Real)
    lbounded, ubounded = isfinite(u.lower), isfinite(u.upper)
    return if lbounded && ubounded
        bma = u.upper - u.lower
        xma = x - u.lower
        xma_over_bma = xma / bma
        [logit(xma_over_bma)], -log(xma_over_bma * (u.upper - x))
    elseif lbounded
        log_xma = log(x - u.lower)
        [log_xma], -log_xma
    elseif ubounded
        log_bmx = log(u.upper - x)
        [log_bmx], -log_bmx
    else
        return zero(x)
    end
end
inverse(u::UntruncateVect) = TruncateOnly(u.lower, u.upper)

# This is the fallback option for all other univariate continuous distributions.
function VectorBijectors.to_linked_vec(d::D.ContinuousUnivariateDistribution)
    return UntruncateVect(minimum(d), maximum(d))
end
function VectorBijectors.from_linked_vec(d::D.ContinuousUnivariateDistribution)
    return TruncateOnly(minimum(d), maximum(d))
end
