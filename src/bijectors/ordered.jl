"""
    OrderedBijector()

A bijector mapping ordered vectors in ℝᵈ to unordered vectors in ℝᵈ.

## See also
- [Stan's documentation](https://mc-stan.org/docs/2_27/reference-manual/ordered-vector.html)
  - Note that this transformation and its inverse are the _opposite_ of in this reference.
"""
struct OrderedBijector <: Bijector end

"""
    ordered(d::Distribution)

Return a `Distribution` whose support are ordered vectors, i.e., vectors with increasingly ordered elements.

This transformation is currently only supported for otherwise unconstrained distributions.
"""
function ordered(d::ContinuousMultivariateDistribution)
    # We're good if the map from unconstrained (in which we apply the ordered bijector)
    # to constrained is monotonically increasing, i.e. order-preserving. In that case,
    # we can form the ordered transformation as `binv ∘ OrderedBijector() ∘ b`.
    # TODO: Add support for monotonically _decreasing_ transformations. This will be the
    # the same as above, but the ordering will be reversed by `binv` so we need to handle this.
    b = bijector(d)
    binv = inverse(b)
    if !is_monotonically_increasing(binv)
        throw(
            ArgumentError(
                "ordered transform is not supported for $d.",
            ),
        )
    end
    return transformed(d, binv ∘ OrderedBijector() ∘ b)
end

with_logabsdet_jacobian(b::OrderedBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(b::OrderedBijector, y::AbstractVecOrMat) = _transform_ordered(y)

function _transform_ordered(y::AbstractVector)
    x = similar(y)
    @assert !isempty(y)

    @inbounds x[1] = y[1]
    @inbounds for i in 2:length(x)
        x[i] = x[i - 1] + exp(y[i])
    end

    return x
end

function _transform_ordered(y::AbstractMatrix)
    x = similar(y)
    @assert !isempty(y)

    @inbounds for j in 1:size(x, 2), i in 1:size(x, 1)
        if i == 1
            x[i, j] = y[i, j]
        else
            x[i, j] = x[i - 1, j] + exp(y[i, j])
        end
    end

    return x
end

transform(ib::Inverse{OrderedBijector}, x::AbstractVecOrMat) = _transform_inverse_ordered(x)
function _transform_inverse_ordered(x::AbstractVector)
    y = similar(x)
    @assert !isempty(y)

    @inbounds y[1] = x[1]
    @inbounds for i in 2:length(y)
        y[i] = log(x[i] - x[i - 1])
    end

    return y
end

function _transform_inverse_ordered(x::AbstractMatrix)
    y = similar(x)
    @assert !isempty(y)

    @inbounds for j in 1:size(y, 2), i in 1:size(y, 1)
        if i == 1
            y[i, j] = x[i, j]
        else
            y[i, j] = log(x[i, j] - x[i - 1, j])
        end
    end

    return y
end

logabsdetjac(b::OrderedBijector, x::AbstractVector) = sum(@view(x[2:end]))
logabsdetjac(b::OrderedBijector, x::AbstractMatrix) = vec(sum(@view(x[2:end, :]); dims=1))
