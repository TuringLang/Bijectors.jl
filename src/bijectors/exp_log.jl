#############
# Exp & Log #
#############

struct Exp{N} <: Bijector{N} end
struct Log{N} <: Bijector{N} end
up1(::Exp{N}) where {N} = Exp{N + 1}()
up1(::Log{N}) where {N} = Log{N + 1}()

Exp() = Exp{0}()
Log() = Log{0}()

(b::Exp{0})(y::Real) = exp(y)
(b::Log{0})(x::Real) = log(x)

(b::Exp{0})(y::AbstractArray{<:Real}) = exp.(y)
(b::Log{0})(x::AbstractArray{<:Real}) = log.(x)

(b::Exp{1})(y::AbstractVector{<:Real}) = exp.(y)
(b::Exp{1})(y::AbstractMatrix{<:Real}) = exp.(y)
(b::Log{1})(x::AbstractVector{<:Real}) = log.(x)
(b::Log{1})(x::AbstractMatrix{<:Real}) = log.(x)

(b::Exp{2})(y::AbstractMatrix{<:Real}) = exp.(y)
(b::Log{2})(x::AbstractMatrix{<:Real}) = log.(x)

(b::Exp{2})(y::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, y)
(b::Log{2})(x::AbstractArray{<:AbstractMatrix{<:Real}}) = map(b, x)

inverse(b::Exp{N}) where {N} = Log{N}()
inverse(b::Log{N}) where {N} = Exp{N}()

logabsdetjac(b::Exp{0}, x::Real) = x
logabsdetjac(b::Exp{0}, x::AbstractVector) = x
logabsdetjac(b::Exp{1}, x::AbstractVector) = sum(x)
logabsdetjac(b::Exp{1}, x::AbstractMatrix) = vec(sum(x; dims = 1))
logabsdetjac(b::Exp{2}, x::AbstractMatrix) = sum(x)
logabsdetjac(b::Exp{2}, x::AbstractArray{<:AbstractMatrix}) = map(x) do x
    logabsdetjac(b, x)
end

logabsdetjac(b::Log{0}, x::Real) = -log(x)
logabsdetjac(b::Log{0}, x::AbstractVector) = .-log.(x)
logabsdetjac(b::Log{1}, x::AbstractVector) = - sum(log, x)
logabsdetjac(b::Log{1}, x::AbstractMatrix) = - vec(sum(log, x; dims = 1))
logabsdetjac(b::Log{2}, x::AbstractMatrix) = - sum(log, x)
logabsdetjac(b::Log{2}, x::AbstractArray{<:AbstractMatrix}) = map(x) do x
    logabsdetjac(b, x)
end
