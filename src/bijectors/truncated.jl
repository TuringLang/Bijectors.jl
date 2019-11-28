#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
struct TruncatedBijector{T} <: Bijector{0}
    lb::T
    ub::T
end

function (b::TruncatedBijector)(x::Real)
    a, b = b.lb, b.ub
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

function (b::TruncatedBijector)(x::AbstractVector{<:Real})
    a, b = b.lb, b.ub
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return @. StatsFuns.logit((x - a) / (b - a))
    elseif lowerbounded
        return log.(x - a)
    elseif upperbounded
        return log.(b - x)
    else
        return x
    end
end

function (ib::Inversed{<:TruncatedBijector})(y::Real)
    a, b = ib.orig.lb, ib.orig.ub
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

function (ib::Inversed{<:TruncatedBijector})(y::AbstractVector{<:Real})
    a, b = ib.orig.lb, ib.orig.ub
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return @. (b - a) * StatsFuns.logistic(y) + a
    elseif lowerbounded
        return @. exp(y) + a
    elseif upperbounded
        return @. b - exp(y)
    else
        return y
    end
end

function logabsdetjac(b::TruncatedBijector, x::Real)
    a, b = b.lb, b.ub
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

function logabsdetjac(b::TruncatedBijector, x::AbstractVector{<:Real})
    a, b = b.lb, b.ub
    lowerbounded, upperbounded = isfinite(a), isfinite(b)
    if lowerbounded && upperbounded
        return @. - log((x - a) * (b - x) / (b - a))
    elseif lowerbounded
        return @. - log(x - a)
    elseif upperbounded
        return @. - log(b - x)
    else
        return zero(x)
    end
end

