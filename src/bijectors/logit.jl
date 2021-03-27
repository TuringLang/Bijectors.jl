######################
# Logit and Logistic #
######################
using StatsFuns: logit, logistic

struct Logit{N, T<:Real} <: Bijector{N}
    a::T
    b::T
end
Logit(a::Real, b::Real) = Logit{0}(a, b)
Logit(a::AbstractArray{<:Real, N}, b::AbstractArray{<:Real, N}) where {N} = Logit{N}(a, b)
function Logit{N}(a, b)
    T = promote_type(typeof(a), typeof(b))
    Logit{N, T}(a, b)
end

# fields are numerical parameters
function Functors.functor(::Type{<:Logit{N}}, x) where N
    function reconstruct_logit(xs)
        T = promote_type(typeof(xs.a), typeof(xs.b))
        return Logit{N,T}(xs.a, xs.b)
    end
    return (a = x.a, b = x.b,), reconstruct_logit
end

up1(b::Logit{N, T}) where {N, T} = Logit{N + 1, T}(b.a, b.b)
# For equality of Logit with Float64 fields to one with Duals
Base.:(==)(b1::Logit, b2::Logit) = b1.a == b2.a && b1.b == b2.b

(b::Logit)(x) = _logit.(x, b.a, b.b)
(b::Logit)(x::AbstractArray{<:AbstractArray}) = map(b, x)
_logit(x, a, b) = logit((x - a) / (b - a))

(ib::Inverse{<:Logit})(y) = _ilogit.(y, ib.orig.a, ib.orig.b)
(ib::Inverse{<:Logit})(x::AbstractArray{<:AbstractArray}) = map(ib, x)
_ilogit(y, a, b) = (b - a) * logistic(y) + a

logabsdetjac(b::Logit{0}, x) = logit_logabsdetjac.(x, b.a, b.b)
logabsdetjac(b::Logit{1}, x::AbstractVector) = sum(logit_logabsdetjac.(x, b.a, b.b))
logabsdetjac(b::Logit{1}, x::AbstractMatrix) = vec(sum(logit_logabsdetjac.(x, b.a, b.b), dims = 1))
logabsdetjac(b::Logit{2}, x::AbstractMatrix) = sum(logit_logabsdetjac.(x, b.a, b.b))
logabsdetjac(b::Logit{2}, x::AbstractArray{<:AbstractMatrix}) = map(x) do x
    logabsdetjac(b, x)
end
logit_logabsdetjac(x, a, b) = -log((x - a) * (b - x) / (b - a))
