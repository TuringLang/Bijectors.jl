# This file handles things like `reshape(dist, newshape...)` or `vec(dist)` by wrapping
# the original bijector and composing it with a reshape operation. It can be inefficient in
# some edge cases (for example, `vec(Normal())` will unwrap and rewrap the value in an
# array), but these cases seem meaningless enough that we can ignore them.

# When reshaping univariate distributions, `original_size` get stored as `()`. If we naively
# use `reshape` we will get a 0-dimensional array out, which is not the same as the scalar
# that `rand(dist)` returns. So we need to use this helper function.
_reshape_or_only(x::AbstractArray, ::Tuple{}) = x[]
_reshape_or_only(x, ::Tuple{}) = x
_reshape_or_only(x::AbstractArray, sz) = reshape(x, sz)
# This method handles the case where we need to 'reshape' a scalar into an array
_reshape_or_only(x, sz) = reshape([x], sz)

"""
    ReshapeWrapper(bijector::Bijector, original_size::Tuple, reshaped_size::Tuple)

Here, `original_size` is equal to `size(dist)`, and `reshaped_size` is the size after
reshaping. The wrapped `bijector` converts a sample from `original_size` to a vector or
linked vector. Thus, ReshapeWrapper must:

- first convert the input from `reshaped_size` to `original_size` via `reshape`
- then apply the wrapped `bijector`.
"""
struct ReshapeWrapper{B,Told<:Tuple,Tnew<:Tuple}
    bijector::B
    original_size::Told
    reshaped_size::Tnew
end
function with_logabsdet_jacobian(r::ReshapeWrapper, rx)
    x = _reshape_or_only(rx, r.original_size)
    return with_logabsdet_jacobian(r.bijector, x)
end
(r::ReshapeWrapper)(rx) = first(with_logabsdet_jacobian(r, rx))
function inverse(r::ReshapeWrapper)
    return InvReshapeWrapper(inverse(r.bijector), r.original_size, r.reshaped_size)
end

"""
    InvReshapeWrapper(inv_bijector::Bijector, original_size::Tuple, reshaped_size::Tuple)

This is the inverse of ReshapeWrapper. It does a similar thing to `ReshapeWrapper`, but in a
different order, since it must apply `inv_bijector` first before reshaping the output.
"""
struct InvReshapeWrapper{B,Told<:Tuple,Tnew<:Tuple}
    inv_bijector::B
    original_size::Told
    reshaped_size::Tnew
end
function with_logabsdet_jacobian(r::InvReshapeWrapper, x)
    rx, ladj = with_logabsdet_jacobian(r.inv_bijector, x)
    rx_reshaped = _reshape_or_only(rx, r.reshaped_size)
    return (rx_reshaped, ladj)
end
(r::InvReshapeWrapper)(x) = first(with_logabsdet_jacobian(r, x))
function inverse(r::InvReshapeWrapper)
    return ReshapeWrapper(inverse(r.inv_bijector), r.original_size, r.reshaped_size)
end

to_vec(d::D.ReshapedDistribution) = ReshapeWrapper(to_vec(d.dist), size(d.dist), size(d))
function from_vec(d::D.ReshapedDistribution)
    return InvReshapeWrapper(from_vec(d.dist), size(d.dist), size(d))
end
vec_length(d::D.ReshapedDistribution) = vec_length(d.dist)
optic_vec(d::D.ReshapedDistribution) = optic_vec(d.dist)
# Need to special-case univariate distributions, because they aren't just reshaped, the
# samples are also arrays instead of scalars (so the optics need one more layer of
# indirection).
function optic_vec(
    ::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:D.UnivariateDistribution}
)
    # TODO(penelopeysm): We assume that the axes of the resulting multivariate (reshaped)
    # distribution are 1-indexed. This is true if you use `reshape()`, but in general we would
    # like to be more aware of distribution axes. See e.g.
    # https://github.com/JuliaStats/Distributions.jl/pull/2009.
    return [AbstractPPL.@opticof(_[1])]
end
function to_linked_vec(d::D.ReshapedDistribution)
    return ReshapeWrapper(to_linked_vec(d.dist), size(d.dist), size(d))
end
function from_linked_vec(d::D.ReshapedDistribution)
    return InvReshapeWrapper(from_linked_vec(d.dist), size(d.dist), size(d))
end
linked_vec_length(d::D.ReshapedDistribution) = linked_vec_length(d.dist)
linked_optic_vec(d::D.ReshapedDistribution) = linked_optic_vec(d.dist)
function linked_optic_vec(
    ::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:D.UnivariateDistribution}
)
    return [AbstractPPL.@opticof(_[1])]
end
