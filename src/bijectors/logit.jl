######################
# Logit and Logistic #
######################
using StatsFuns: logit, logistic

struct Logit{T<:Real} <: Bijector{0}
    a::T
    b::T
end

(b::Logit)(x::Real) = _logit(x, b.a, b.b)
(b::Logit)(x) = _logit.(x, b.a, b.b)
_logit(x, a, b) = logit((x - a) / (b - a))

(ib::Inverse{<:Logit})(y::Real) = _ilogit(y, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit})(y) = _ilogit.(y, ib.orig.a, ib.orig.b)
_ilogit(y, a, b) = (b - a) * logistic(y) + a

logabsdetjac(b::Logit, x::Real) = logit_logabsdetjac(x, b.a, b.b)
logabsdetjac(b::Logit, x) = logit_logabsdetjac.(x, b.a, b.b)
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))