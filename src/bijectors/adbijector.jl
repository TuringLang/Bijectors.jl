"""
    ADBijector(d::Distribution)
    ADBijector{<:ADBackend, D, N}(d::Distribution)

This a subtype of `Bijector{N}` that uses `link` and `invlink` to compute the 
transformations, and `AD` to compute the `jacobian` and `logabsdetjac`.
"""
struct ADBijector{AD, D, N} <: Bijector{N} where {D<:Distribution}
    dist::D
end
function ADBijector(dist::D) where {D<:UnivariateDistribution}
    ADBijector{ADBackend(), D, 0}(dist)
end
function ADBijector(dist::D) where {D<:MultivariateDistribution}
    ADBijector{ADBackend(), D, 1}(dist)
end
function ADBijector(dist::D) where {D<:MatrixDistribution}
    ADBijector{ADBackend(), D, 2}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
(b::ADBijector)(x) = link(b.dist, x)
(ib::Inverse{<:ADBijector})(y) = invlink(ib.orig.dist, y)

struct SingularJacobianException{B<:Bijector} <: Exception
    b::B
end
function Base.showerror(io::IO, e::SingularJacobianException)
    print(io, "jacobian of $(e.b) is singular")
end

# concrete implementations with optional dependencies ForwardDiff and Tracker
function jacobian end

# TODO: allow batch-computation, especially for univariate case?
"Computes the absolute determinant of the Jacobian of the inverse-transformation."
function logabsdetjac(b::ADBijector, x::Real)
    res = log(abs(jacobian(b, x)))
    return isfinite(res) ? res : throw(SingularJacobianException(b))
end

function logabsdetjac(b::ADBijector, x::AbstractVector{<:Real})
    fact = lu(jacobian(b, x), check=false)
    return issuccess(fact) ? logabsdet(fact)[1] : throw(SingularJacobianException(b))
end
