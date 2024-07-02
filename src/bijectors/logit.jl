######################
# Logit and Logistic #
######################
struct Logit{T1,T2} <: Bijector
    a::T1
    b::T2
end

Functors.@functor Logit

# For equality of Logit with Float64 fields to one with Duals
Base.:(==)(b1::Logit, b2::Logit) = b1.a == b2.a && b1.b == b2.b

# Evaluation
_logit(x, a, b) = LogExpFunctions.logit((x - a) / (b - a))
transform(b::Logit, x) = _logit.(x, b.a, b.b)

# Inverse
_ilogit(y, a, b) = (b - a) * LogExpFunctions.logistic(y) + a

transform(ib::Inverse{<:Logit}, y) = _ilogit.(y, ib.orig.a, ib.orig.b)

# `logabsdetjac`
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))
logabsdetjac(b::Logit, x) = sum(logit_logabsdetjac.(x, b.a, b.b))

# `with_logabsdet_jacobian`
function with_logabsdet_jacobian(b::Logit, x)
    return _logit.(x, b.a, b.b), sum(logit_logabsdetjac.(x, b.a, b.b))
end

is_monotonically_increasing(::Logit) = true
