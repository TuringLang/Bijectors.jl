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
    ReshapeWrapper(reshaped_size::Tuple, original_size::Tuple, bijector::Bijector)

Here, `original_size` is equal to `size(dist)`, and `reshaped_size` is the size after
reshaping. The wrapped `bijector` converts a sample from `original_size` to a vector or
linked vector. Thus, ReshapeWrapper must:

- first convert the input from `reshaped_size` to `original_size` via `reshape`
- then apply the wrapped `bijector`.
"""
struct ReshapeWrapper{N1,N2,T1<:NTuple{N1,Int},T2<:NTuple{N2,Int},B}
    reshaped_size::T1
    original_size::T2
    bijector::B
end
function with_logabsdet_jacobian(
    r::ReshapeWrapper{N1}, rx::AbstractArray{T,N1}
) where {T,N1}
    x = _reshape_or_only(rx, r.original_size)
    return with_logabsdet_jacobian(r.bijector, x)
end
function (r::ReshapeWrapper{N1})(rx::AbstractArray{T,N1}) where {T,N1}
    return first(with_logabsdet_jacobian(r, rx))
end
function inverse(r::ReshapeWrapper)
    return InvReshapeWrapper(r.reshaped_size, r.original_size, inverse(r.bijector))
end

"""
    InvReshapeWrapper(reshaped_size::Tuple, original_size::Tuple, inv_bijector::Bijector)

This is the inverse of ReshapeWrapper. It does a similar thing to `ReshapeWrapper`, but in a
different order, since it must apply `inv_bijector` first before reshaping the output.
"""
struct InvReshapeWrapper{N1,N2,T1<:NTuple{N1,Int},T2<:NTuple{N2,Int},B}
    reshaped_size::T1
    original_size::T2
    inv_bijector::B
end
function with_logabsdet_jacobian(r::InvReshapeWrapper, x::AbstractVector)
    rx, ladj = with_logabsdet_jacobian(r.inv_bijector, x)
    rx_reshaped = _reshape_or_only(rx, r.reshaped_size)
    return (rx_reshaped, ladj)
end
(r::InvReshapeWrapper)(x::AbstractVector) = first(with_logabsdet_jacobian(r, x))
function inverse(r::InvReshapeWrapper)
    return ReshapeWrapper(r.reshaped_size, r.original_size, inverse(r.inv_bijector))
end

# Need some special cases for optics.
const ReshapedUnivariateDistribution = D.ReshapedDistribution{
    <:Any,<:D.ValueSupport,<:D.UnivariateDistribution
}

to_vec(d::D.ReshapedDistribution) = ReshapeWrapper(size(d), size(d.dist), to_vec(d.dist))
function from_vec(d::D.ReshapedDistribution)
    return InvReshapeWrapper(size(d), size(d.dist), from_vec(d.dist))
end
vec_length(d::D.ReshapedDistribution) = vec_length(d.dist)

function to_linked_vec(d::D.ReshapedDistribution)
    return ReshapeWrapper(size(d), size(d.dist), to_linked_vec(d.dist))
end
function from_linked_vec(d::D.ReshapedDistribution)
    return InvReshapeWrapper(size(d), size(d.dist), from_linked_vec(d.dist))
end
linked_vec_length(d::D.ReshapedDistribution) = linked_vec_length(d.dist)

# optic_vec requires some care. We can't just reuse the original distribution's optics,
# i.e., `optic_vec(d) = optic_vec(d.dist)` because the axes may have changed due to
# reshaping. In some cases it might just happen to work (e.g. if `d.dist` is a multivariate
# distribution, `optic_vec(d.dist)` would return [_[1], _[2], ...] which would work on any
# AbstractArray because of linear indexing. However, that isn't general.
#
# Broadly speaking, we need to map the original distribution's optics through the reshape
# operation. That is, if `optic_vec(d.dist)` returns `[_[i1...], _[i2...], ...]` where `i1`
# and `i2` are tuples of indices into the original distribution's array, we need to return
# `[_[j1...], _[j2...], ...]` where `j1` is the indices that `i1` would be mapped to by the
# reshape.
#
# We can probably safely assume (for now) that `optic_vec(d.dist)` doesn't return anything
# that has a more complicated structure than an array index. For example, `d.dist` couldn't
# be something like LKJCholesky because you can't call `reshape(LKJCholesky(...), ...)`
# anyway.
function optic_vec(d::D.ReshapedDistribution)
    original_optics = optic_vec(d.dist)
    linear_indices_original = LinearIndices(size(d.dist))
    cartesian_indices_reshaped = CartesianIndices(size(d))
    mapped_optics = map(original_optics) do opt
        if opt isa AbstractPPL.Index
            # Don't know how to generally handle this yet. Probably not an issue yet
            # because Distributions.jl is not fancy enough to have complicated axes.
            if !isempty(opt.kw)
                error("optic_vec for ReshapedDistribution only supports simple Index optics")
            end
            # Map the indices through the reshape
            linear_index = linear_indices_original[opt.ix...]
            new_cartesian_index = cartesian_indices_reshaped[linear_index]
            return AbstractPPL.Index(new_cartesian_index.I, (;), opt.child)
        else
            error("optic_vec for ReshapedDistribution only supports Index optics")
        end
    end
    return mapped_optics
end
# If `d.dist` is univariate that is a special case because `optic_vec(d.dist)` would return
# [Iden]`. In that case we need to tack on the array indices.
function optic_vec(d::ReshapedUnivariateDistribution)
    # size(d) should be a tuple that contains only 1's, so we can just reuse it
    return [AbstractPPL.Index(size(d), (;))]
end

# linked_optic_vec is the same...
function linked_optic_vec(d::D.ReshapedDistribution)
    original_optics = linked_optic_vec(d.dist)
    linear_indices_original = LinearIndices(size(d.dist))
    cartesian_indices_reshaped = CartesianIndices(size(d))
    mapped_optics = map(original_optics) do opt
        if opt isa AbstractPPL.Index
            if !isempty(opt.kw)
                error("optic_vec for ReshapedDistribution only supports simple Index optics")
            end
            linear_index = linear_indices_original[opt.ix...]
            new_cartesian_index = cartesian_indices_reshaped[linear_index]
            return AbstractPPL.Index(new_cartesian_index.I, (;), opt.child)
        elseif opt === nothing
            # ... but we just need to make sure to forward any `nothing`s.
            return nothing
        else
            error("optic_vec for ReshapedDistribution only supports Index optics")
        end
    end
    return mapped_optics
end
function linked_optic_vec(d::ReshapedUnivariateDistribution)
    return [AbstractPPL.Index(size(d), (;))]
end
