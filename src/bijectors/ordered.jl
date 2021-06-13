"""
    OrderedBijector()

A bijector mapping ordered vectors in ℝᵈ to an unordered vectors in ℝᵈ.

## See also
- [Stan's documentation](https://mc-stan.org/docs/2_27/reference-manual/ordered-vector.html)
  - Note that this transformation and its inverse are the _opposite_ of in this reference.
"""
struct OrderedBijector <: Bijector{1} end

"""
    ordered(d::Distribution)

Returns a `Distribution` whose domain is now ordered vectors.
"""
ordered(d::Distribution) = Bijectors.transformed(d, OrderedBijector())

(b::OrderedBijector)(y::AbstractVecOrMat) = _transform_ordered(y)

function _transform_ordered(y::AbstractVector)
    x = similar(y)
    x[1] = y[1]
    @inbounds for i = 2:size(x, 1)
        x[i] = x[i - 1] + exp(y[i])
    end

    return x
end

function _transform_ordered(y::AbstractMatrix)
    x = similar(y)
    x[1, :] = y[1, :]
    @inbounds for i = 2:size(x, 1)
        x[i, :] = x[i - 1, :] + exp.(y[i, :])
    end

    return x
end

(ib::Inverse{<:OrderedBijector})(x::AbstractVecOrMat) = _transform_inverse_ordered(x)

function _transform_inverse_ordered(x::AbstractVector)
    y = similar(x)
    y[1] = x[1]
    @. y[2:end] = log(x[2:end] - x[1:end - 1])

    return y
end

function _transform_inverse_ordered(x::AbstractMatrix)
    y = similar(x)
    y[1, :] = x[1, :]
    @. y[2:end, :] = log(x[2:end, :] - x[1:end - 1, :])

    return y
end

logabsdetjac(b::OrderedBijector, x::AbstractVector) = sum(x[2:end])
logabsdetjac(b::OrderedBijector, x::AbstractMatrix) = vec(sum(x[2:end, :]; dims = 1))
