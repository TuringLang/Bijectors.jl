# TODO: Deprecate?
"""
    DistributionBijector(d::Distribution)
    DistributionBijector{<:ADBackend, D}(d::Distribution)

This is the default `Bijector` for a distribution. 

It uses `link` and `invlink` to compute the transformations, and `AD` to compute
the `jacobian` and `logabsdetjac`.
"""
struct DistributionBijector{AD, D, N} <: ADBijector{AD, N} where {D<:Distribution}
    dist::D
end
function DistributionBijector(dist::D) where {D<:UnivariateDistribution}
    DistributionBijector{ADBackend(), D, 0}(dist)
end
function DistributionBijector(dist::D) where {D<:MultivariateDistribution}
    DistributionBijector{ADBackend(), D, 1}(dist)
end
function DistributionBijector(dist::D) where {D<:MatrixDistribution}
    DistributionBijector{ADBackend(), D, 2}(dist)
end

# Simply uses `link` and `invlink` as transforms with AD to get jacobian
(b::DistributionBijector)(x) = link(b.dist, x)
(ib::Inverse{<:DistributionBijector})(y) = invlink(ib.orig.dist, y)
