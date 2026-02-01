# OrderStatistic can only ever wrap univariate distributions so these can just delegate to
# the underlying distribution.
to_vec(d::D.OrderStatistic) = to_vec(d.dist)
from_vec(d::D.OrderStatistic) = from_vec(d.dist)
to_linked_vec(d::D.OrderStatistic) = to_linked_vec(d.dist)
from_linked_vec(d::D.OrderStatistic) = from_linked_vec(d.dist)
# We don't need to implement the other methods as OrderStatistic is a subtype of
# UnivariateDistribution, so we can just use the default methods.

# For JointOrderStatistics, we need to map the original bijector over each element. That
# gives us an ordered vector, but we are not done there: we need to then map that ordered
# vector back to a regular (unordered) vector. This uses something similar to
# OrderedBijector(), but we reimplement it here to avoid extra allocations.
struct JointOrderWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::JointOrderWrap)(x::AbstractVector) = first(with_logabsdet_jacobian(w, x))
function with_logabsdet_jacobian(m::JointOrderWrap, x::AbstractVector{T}) where {T<:Real}
    # `x` is always an ordered vector. Sometimes, mapping m.bijector over x doesn't give
    # an ordered vector: it could give a *reverse* ordered vector, if m.bijector performs
    # a sign flip (i.e., is monotonically decreasing). In that case, we need to undo the
    # sign flip to get back to the original ordering.
    s = B.is_monotonically_decreasing(m.bijector) ? -1 : 1
    logjac = zero(T)
    y = similar(x)
    for i in eachindex(x)
        yi, lj = with_logabsdet_jacobian(m.bijector, x[i])
        y[i] = s * yi
        logjac += lj
    end
    # Now `y` will definitely be ordered. To transform this to an unordered vector, we
    # see that y[1] is already fine (it ranges from -Inf to Inf), but y[2] has a lower
    # bound of y[1]. So we need to shift it down by y[1] and take the logarithm.
    # Similarly, y[3] has a lower bound of y[2], etc. etc.
    if length(x) > 1
        shift = y[1]
        for i in eachindex(y)[2:end]
            temp = y[i]
            y[i] = log(temp - shift)
            logjac -= log(temp - shift)
            shift = temp
        end
    end
    return y, logjac
end
inverse(m::JointOrderWrap) = InverseJointOrderWrap(inverse(m.bijector))

struct InverseJointOrderWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::InverseJointOrderWrap)(y::AbstractVector) = first(with_logabsdet_jacobian(w, y))
function with_logabsdet_jacobian(
    m::InverseJointOrderWrap, y::AbstractVector{T}
) where {T<:Real}
    # First, we need to undo the logarithmic transformations to get back to the ordered
    # vector.
    logjac = zero(T)
    x = copy(y)
    if length(y) > 1
        for i in eachindex(y)[2:end]
            temp = x[i]
            x[i] = exp(temp) + x[i - 1]
            logjac += temp
        end
    end
    s = B.is_monotonically_decreasing(m.bijector) ? -1 : 1
    # Now `x` is an ordered vector. We need to apply the signflip if necessary, and then
    # map the inner bijector.
    for i in eachindex(x)
        xi, lj = with_logabsdet_jacobian(m.bijector, s * x[i])
        x[i] = xi
        logjac += lj
    end
    return x, logjac
end
inverse(m::InverseJointOrderWrap) = JointOrderWrap(inverse(m.bijector))

# Here, because `d.dist` isa UnivariateDistribution, `to_vec(d.dist)` or `from_vec(d.dist)`
# returns an OnlyWrap or a VectWrap, whose inner bijector maps scalars to scalars.
# We can then rewrap that inner bijector into a JointOrderWrap to get the desired behavior.
to_vec(::D.JointOrderStatistics) = TypedIdentity()
function to_linked_vec(d::D.JointOrderStatistics)
    return JointOrderWrap(get_inner_bijector(to_linked_vec(d.dist)))
end
from_vec(::D.JointOrderStatistics) = TypedIdentity()
function from_linked_vec(d::D.JointOrderStatistics)
    return InverseJointOrderWrap(get_inner_bijector(from_linked_vec(d.dist)))
end
# Since D.JointOrderStatistics is a subtype of MultivariateDistribution, we can use the
# default definitions for vec_length and optic_vec.
linked_vec_length(d::D.JointOrderStatistics) = vec_length(d)
# Technically, the first element can be @opticof(_[1]) so this is not technically correct.
linked_optic_vec(d::D.JointOrderStatistics) = fill(nothing, vec_length(d))
