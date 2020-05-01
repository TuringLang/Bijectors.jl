######################
# Logit and Logistic #
######################
using StatsFuns: logit, logistic

struct Logit{N, T<:Real} <: Bijector{N}
    a::T
    b::T
end
function Logit(a, b)
    T = promote_type(typeof(a), typeof(b))
    Logit{0, T}(a, b)
end
up1(b::Logit{N, T}) where {N, T} = Logit{N + 1, T}(b.a, b.b)

(b::Logit{0})(x::Real) = _logit(x, b.a, b.b)
(b::Logit{0})(x) = _logit.(x, b.a, b.b)
(b::Logit{1})(x::AbstractVector) = _logit.(x, b.a, b.b)
(b::Logit{1})(x::AbstractMatrix) = _logit.(x, b.a, b.b)
(b::Logit{2})(x::AbstractMatrix) = _logit.(x, b.a, b.b)
(b::Logit{2})(x::AbstractArray{<:AbstractMatrix}) = map(b, x)
_logit(x, a, b) = logit((x - a) / (b - a))

(ib::Inverse{<:Logit{0}})(y::Real) = _ilogit(y, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit{0}})(y) = _ilogit.(y, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit{1}})(x::AbstractVecOrMat) = _ilogit.(x, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit{2}})(x::AbstractMatrix) = _ilogit.(x, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit{2}})(x::AbstractArray{<:AbstractMatrix}) = map(ib, x)
_ilogit(y, a, b) = (b - a) * logistic(y) + a

logabsdetjac(b::Logit{0}, x::Real) = logit_logabsdetjac(x, b.a, b.b)
logabsdetjac(b::Logit{0}, x) = logit_logabsdetjac.(x, b.a, b.b)
logabsdetjac(b::Logit{1}, x::AbstractVector) = sum(logit_logabsdetjac.(x, b.a, b.b))
logabsdetjac(b::Logit{1}, x::AbstractMatrix) = vec(sum(logit_logabsdetjac.(x, b.a, b.b), dims = 1))
logabsdetjac(b::Logit{2}, x::AbstractMatrix) = sum(logit_logabsdetjac.(x, b.a, b.b))
logabsdetjac(b::Logit{2}, x::AbstractArray{<:AbstractMatrix}) = map(x) do x
    logabsdetjac(b, x)
end
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))