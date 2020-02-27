######################
# Logit and Logistic #
######################
using StatsFuns: logit, logistic

struct Logit{T<:Real} <: Bijector{0}
    a::T
    b::T
end

(b::Logit)(x::Real) = @. logit((x - b.a) / (b.b - b.a))
(b::Logit)(x) = b.(x)

(ib::Inverse{<:Logit})(y::Real) = (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a
(ib::Inverse{<:Logit})(y) = ib.(y)

_logabsdetjac(b::Logit, x::Real) = -log((x - b.a) * (b.b - x) / (b.b - b.a))
logabsdetjac(b::Logit, x) = @. _logabsdetjac(b, x)
