######################
# Logit and Logistic #
######################
struct Logit{T} <: Bijector
    a::T
    b::T
end

Functors.@functor Logit

# For equality of Logit with Float64 fields to one with Duals
Base.:(==)(b1::Logit, b2::Logit) = b1.a == b2.a && b1.b == b2.b

# TODO: Implement `forward` and batched versions.

# Evaluation
_logit(x, a, b) = LogExpFunctions.logit((x - a) / (b - a))
transform_single(b::Logit, x) = _logit.(x, b.a, b.b)
function transform_multiple(b::Logit, xs::Batching.ArrayBatch{<:Real})
    return batch_like(xs, _logit.(Batching.value(xs)))
end

# Inverse
_ilogit(y, a, b) = (b - a) * LogExpFunctions.logistic(y) + a

transform_single(ib::Inverse{<:Logit}, y) = _ilogit.(y, ib.orig.a, ib.orig.b)
function transform_multiple(ib::Inverse{<:Logit}, ys::Batching.ArrayBatch)
    return batch_like(ys, _ilogit.(Batching.value(ys)))
end

# `logabsdetjac`
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))
logabsdetjac_single(b::Logit, x) = sum(logit_logabsdetjac.(x, b.a, b.b))
