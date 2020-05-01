#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
struct TruncatedBijector{T1, T2} <: Bijector{0}
    lb::T1
    ub::T2
end

function (b::TruncatedBijector)(x::Real)
    a, b = b.lb, b.ub
    truncated_link(_clamp(x, a, b), a, b)
end 
function (b::TruncatedBijector)(x::AbstractArray{<:Real})
    a, b = b.lb, b.ub
    truncated_link.(_clamp.(x, a, b), a, b)
end
function truncated_link(x::Real, a, b)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return StatsFuns.logit((x - a) / (b - a))
    elseif lowerbounded
        return log(x - a)
    elseif upperbounded
        return log(b - x)
    else
        return x
    end
end

function (ib::Inverse{<:TruncatedBijector})(y::Real)
    a, b = ib.orig.lb, ib.orig.ub
    _clamp(truncated_invlink(y, a, b), a, b)
end
function (ib::Inverse{<:TruncatedBijector})(y::AbstractArray{<:Real})
    a, b = ib.orig.lb, ib.orig.ub
    _clamp.(truncated_invlink.(y, a, b), a, b)
end
function truncated_invlink(y, a, b)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return (b - a) * StatsFuns.logistic(y) + a
    elseif lowerbounded
        return exp(y) + a
    elseif upperbounded
        return b - exp(y)
    else
        return y
    end
end

function logabsdetjac(b::TruncatedBijector, x::Real)
    a, b = b.lb, b.ub
    truncated_logabsdetjac(_clamp(x, a, b), a, b)
end
function logabsdetjac(b::TruncatedBijector, x::AbstractArray{<:Real})
    a, b = b.lb, b.ub
    truncated_logabsdetjac.(_clamp.(x, a, b), a, b)
end
function truncated_logabsdetjac(x, a, b)
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return - log((x - a) * (b - x) / (b - a))
    elseif lowerbounded
        return - log(x - a)
    elseif upperbounded
        return - log(b - x)
    else
        return zero(x)
    end
end
