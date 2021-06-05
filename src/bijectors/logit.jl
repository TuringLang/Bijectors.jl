######################
# Logit and Logistic #
######################
using StatsFuns: logit, logistic

struct Logit{T} <: Bijector
    a::T
    b::T
end

Functors.@functor Logit

# For equality of Logit with Float64 fields to one with Duals
Base.:(==)(b1::Logit, b2::Logit) = b1.a == b2.a && b1.b == b2.b

transform(b::Logit, x) = _logit.(x, b.a, b.b)
_logit(x, a, b) = logit((x - a) / (b - a))

transform(ib::Inverse{<:Logit}, y) = _ilogit.(y, ib.orig.a, ib.orig.b)
_ilogit(y, a, b) = (b - a) * logistic(y) + a

logabsdetjac(b::Logit, x) = sum(logit_logabsdetjac.(x, b.a, b.b))
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))
