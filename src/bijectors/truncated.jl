#######################################################
# Constrained to unconstrained distribution bijectors #
#######################################################
struct TruncatedBijector{N, T1, T2} <: Bijector{N}
    lb::T1
    ub::T2
end

@functor TruncatedBijector

TruncatedBijector(lb, ub) = TruncatedBijector{0}(lb, ub)
function TruncatedBijector{N}(lb::T1, ub::T2) where {N, T1, T2}
    return TruncatedBijector{N, T1, T2}(lb, ub)
end
up1(b::TruncatedBijector{N}) where {N} = TruncatedBijector{N + 1}(b.lb, b.ub)

function Base.:(==)(b1::TruncatedBijector, b2::TruncatedBijector)
    return b1.lb == b2.lb && b1.ub == b2.ub
end

function (b::TruncatedBijector{0})(x::Real)
    a, b = b.lb, b.ub
    truncated_link(_clamp(x, a, b), a, b)
end 
function (b::TruncatedBijector{0})(x::AbstractArray{<:Real})
    a, b = b.lb, b.ub
    truncated_link.(_clamp.(x, a, b), a, b)
end
function (b::TruncatedBijector{1})(x::AbstractVecOrMat{<:Real})
    a, b = b.lb, b.ub
    if a isa AbstractVector
        @assert b isa AbstractVector
        maporbroadcast(x, a, b) do x, a, b
            truncated_link(_clamp(x, a, b), a, b)
        end
    else
        truncated_link.(_clamp.(x, a, b), a, b)
    end
end
function (b::TruncatedBijector{2})(x::AbstractMatrix{<:Real})
    a, b = b.lb, b.ub
    if a isa AbstractMatrix
        @assert b isa AbstractMatrix
        maporbroadcast(x, a, b) do x, a, b
            truncated_link(_clamp(x, a, b), a, b)
        end
    else
        truncated_link.(_clamp.(x, a, b), a, b)
    end
end
(b::TruncatedBijector{2})(x::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, x)
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

function (ib::Inverse{<:TruncatedBijector{0}})(y::Real)
    a, b = ib.orig.lb, ib.orig.ub
    _clamp(truncated_invlink(y, a, b), a, b)
end
function (ib::Inverse{<:TruncatedBijector{0}})(y::AbstractArray{<:Real})
    a, b = ib.orig.lb, ib.orig.ub
    _clamp.(truncated_invlink.(y, a, b), a, b)
end
function (ib::Inverse{<:TruncatedBijector{1}})(y::AbstractVecOrMat{<:Real})
    a, b = ib.orig.lb, ib.orig.ub
    if a isa AbstractVector
        @assert b isa AbstractVector
        maporbroadcast(y, a, b) do y, a, b
            _clamp(truncated_invlink(y, a, b), a, b)
        end
    else
        _clamp.(truncated_invlink.(y, a, b), a, b)
    end
end
function (ib::Inverse{<:TruncatedBijector{2}})(y::AbstractMatrix{<:Real})
    a, b = ib.orig.lb, ib.orig.ub
    if a isa AbstractMatrix
        @assert b isa AbstractMatrix
        return maporbroadcast(y, a, b) do y, a, b
            _clamp(truncated_invlink(y, a, b), a, b)
        end
    else
        return _clamp.(truncated_invlink.(y, a, b), a, b)
    end
end
(ib::Inverse{<:TruncatedBijector{2}})(y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(ib, y)
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

function logabsdetjac(b::TruncatedBijector{0}, x::Real)
    a, b = b.lb, b.ub
    truncated_logabsdetjac(_clamp(x, a, b), a, b)
end
function logabsdetjac(b::TruncatedBijector{0}, x::AbstractArray{<:Real})
    a, b = b.lb, b.ub
    truncated_logabsdetjac.(_clamp.(x, a, b), a, b)
end
function logabsdetjac(b::TruncatedBijector{1}, x::AbstractVector{<:Real})
    a, b = b.lb, b.ub
    sum(truncated_logabsdetjac.(_clamp.(x, a, b), a, b))
end
function logabsdetjac(b::TruncatedBijector{1}, x::AbstractMatrix{<:Real})
    a, b = b.lb, b.ub
    vec(sum(truncated_logabsdetjac.(_clamp.(x, a, b), a, b), dims = 1))
end
function logabsdetjac(b::TruncatedBijector{2}, x::AbstractMatrix{<:Real})
    a, b = b.lb, b.ub
    sum(truncated_logabsdetjac.(_clamp.(x, a, b), a, b))
end
function logabsdetjac(b::TruncatedBijector{2}, x::AbstractArray{<:AbstractMatrix{<:Real}})
    map(x) do x
        logabsdetjac(b, x)
    end
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
