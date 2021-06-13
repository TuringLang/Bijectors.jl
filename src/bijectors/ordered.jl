struct OrderedBijector <: Bijector{1} end

"""
    ordered(d::Distribution)

TODO
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
