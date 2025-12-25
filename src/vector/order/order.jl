# OrderStatistic can only ever wrap univariate distributions so these can just delegate to
# the underlying distribution.
to_vec(d::D.OrderStatistic) = to_vec(d.dist)
from_vec(d::D.OrderStatistic) = from_vec(d.dist)
to_linked_vec(d::D.OrderStatistic) = to_linked_vec(d.dist)
from_linked_vec(d::D.OrderStatistic) = from_linked_vec(d.dist)
# We don't need to implement the other methods as OrderStatistic is a subtype of
# UnivariateDistribution, so we can just use the default methods.

# For JointOrderStatistics, we need to essentially map the original bijector over each
# element.
struct MapWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::MapWrap)(x::AbstractVector) = first(with_logabsdet_jacobian(w, x))
function with_logabsdet_jacobian(::MapWrap{TypedIdentity}, x::AbstractVector)
    return with_logabsdet_jacobian(TypedIdentity(), x)
end
function with_logabsdet_jacobian(m::MapWrap, x::AbstractVector{T}) where {T<:Real}
    logjac = zero(T)
    y = similar(x)
    for i in eachindex(x)
        y[i], lj = with_logabsdet_jacobian(m.bijector, x[i])
        logjac += lj
    end
    return y, logjac
end
# Needed to avoid method ambiguity with the above
function with_logabsdet_jacobian(
    ::MapWrap{TypedIdentity}, x::AbstractVector{T}
) where {T<:Real}
    return with_logabsdet_jacobian(TypedIdentity(), x)
end

# Here, because `d.dist` isa UnivariateDistribution, `to_vec(d.dist)` or `from_vec(d.dist)`
# returns an OnlyWrap or a VectWrap, whose inner bijector maps scalars to scalars.
# We can then rewrap that inner bijector into a MapWrap to get the desired behavior.
to_vec(d::D.JointOrderStatistics) = MapWrap(get_inner_bijector(to_vec(d.dist)))
function to_linked_vec(d::D.JointOrderStatistics)
    return MapWrap(get_inner_bijector(to_linked_vec(d.dist)))
end
from_vec(d::D.JointOrderStatistics) = MapWrap(get_inner_bijector(from_vec(d.dist)))
function from_linked_vec(d::D.JointOrderStatistics)
    return MapWrap(get_inner_bijector(from_linked_vec(d.dist)))
end
# Since D.JointOrderStatistics is a subtype of MultivariateDistribution, we can use the
# default definitions for vec_length and optic_vec, and just forward linked_vec_length and
# linked_optic_vec.
linked_vec_length(d::D.JointOrderStatistics) = vec_length(d)
linked_optic_vec(d::D.JointOrderStatistics) = optic_vec(d)
