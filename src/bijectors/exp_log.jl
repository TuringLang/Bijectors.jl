#############
# Exp & Log #
#############

struct Exp{N} <: Bijector{N} end
struct Log{N} <: Bijector{N} end

Exp() = Exp{0}()
Log() = Log{0}()

(b::Exp)(y) = @. exp(y)
(b::Log)(x) = @. log(x)

inv(b::Exp{N}) where {N} = Log{N}()
inv(b::Log{N}) where {N} = Exp{N}()

logabsdetjac(b::Exp{0}, x::Real) = x
logabsdetjac(b::Exp{0}, x::AbstractVector) = x
logabsdetjac(b::Exp{1}, x::AbstractVector) = sum(x)
logabsdetjac(b::Exp{1}, x::AbstractMatrix) = vec(sum(x; dims = 1))

logabsdetjac(b::Log{0}, x::Real) = -log(x)
logabsdetjac(b::Log{0}, x::AbstractVector) = -log.(x)
logabsdetjac(b::Log{1}, x::AbstractVector) = - sum(log.(x))
logabsdetjac(b::Log{1}, x::AbstractMatrix) = - vec(sum(log.(x); dims = 1))
