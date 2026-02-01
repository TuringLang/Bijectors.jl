# Generic definitions for matrix distributions.

# Somehow, ChangesOfVariables doesn't have a logjac implemented for `vec`, so we need to
# wrap it.
struct Vec end
(::Vec)(x::AbstractArray) = vec(x)
function with_logabsdet_jacobian(::Vec, x::AbstractArray{N,T}) where {N,T<:Number}
    return vec(x), zero(T)
end
function with_logabsdet_jacobian(::Vec, x::AbstractArray)
    return vec(x), 0.0
end

struct Reshape{N,T<:NTuple{N,Int}}
    size::T
end
(r::Reshape)(x::AbstractArray) = reshape(x, r.size)
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray{N,T}) where {N,T<:Number}
    return reshape(x, r.size), zero(T)
end
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray)
    return reshape(x, r.size), 0.0
end

to_vec(::D.MatrixDistribution) = Vec()
from_vec(d::D.MatrixDistribution) = Reshape(size(d))
vec_length(d::D.MatrixDistribution) = prod(size(d))
function optic_vec(d::D.MatrixDistribution)
    return map(c -> AbstractPPL.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end
