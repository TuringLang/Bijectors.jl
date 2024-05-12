struct SignFlip <: Bijector end

with_logabsdet_jacobian(::SignFlip, x) = -x, zero(eltype(x))
inverse(::SignFlip) = SignFlip()
output_size(::SignFlip, dim) = dim
is_monotonically_increasing(::SignFlip) = false
is_monotonically_decreasing(::SignFlip) = false

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
    # Similarly, if we're working with monotonically decreasing maps, we can do the same
    # but with the addition of a sign flip before and after the ordered bijector.
    b = bijector(d)
    binv = inverse(b)
    if is_monotonically_decreasing(binv)
        ordered_b = binv ∘ SignFlip() ∘ OrderedBijector() ∘ SignFlip() ∘ b
    elseif is_monotonically_increasing(binv)
        ordered_b = binv ∘ OrderedBijector() ∘ b
    else
        throw(ArgumentError("ordered transform is currently not supported for $d."))
    end

    return transformed(d, ordered_b)
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

# Need a custom distribution type to handle this properly.
struct OrderedDistribution{B,D<:ContinuousMultivariateDistribution} <:
       ContinuousMultivariateDistribution
    dist::D
    bijector::B
end

bijector(d::OrderedDistribution) = d.bijector

Distributions.logpdf(d::OrderedDistribution, x) = Distributions.logpdf(d.dist, x)
function Distributions._logpdf(d::OrderedDistribution, x::AbstractVector{<:Real})
    issorted(x) || return -eltype(d)(Inf)
    return Distributions.logpdf(d.dist, x)
end
Base.length(d::OrderedDistribution) = length(d.dist)

"""
    marginal_dist(d::MultivariateDistribution, i::Int)

Return the marginal distribution of the `i`-th component of the multivariate
distribution `d`.
"""
marginal_dist(d::Distributions.Product, i::Int) = d.v[i]
function marginal_dist(d::AbstractMvNormal, i::Int)
    return Normal(mean(d)[i], sqrt(var(d)[i]))
end

function Distributions._rand!(
    rng::AbstractRNG, d::OrderedDistribution, x::AbstractVector{<:Real}
)
    # Rejection sampling.
    x[1] = rand(rng, marginal_dist(d.dist, 1))
    for i in 2:length(x)
        x_prev = x[i - 1]
        d_i = marginal_dist(d.dist, i)
        success = false
        while !success
            x[i] = rand(rng, d_i)
            success |= x[i] >= x_prev
        end
    end
    return x
end
