struct SignFlip <: Bijector end

with_logabsdet_jacobian(::SignFlip, x) = -x, zero(eltype(x))
inverse(::SignFlip) = SignFlip()
output_size(::SignFlip, dim) = dim
is_monotonically_increasing(::SignFlip) = false
is_monotonically_decreasing(::SignFlip) = true

"""
    OrderedBijector()

A bijector mapping unordered vectors in ℝᵈ to ordered vectors in ℝᵈ.

## See also
- [Stan's documentation](https://mc-stan.org/docs/2_27/reference-manual/ordered-vector.html)
  - Note that this transformation and its inverse are the _opposite_ of in this reference.
"""
struct OrderedBijector <: Bijector end

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
"""
    OrderedDistribution

Wraps a distribution to restrict its support to the subspace of ordered vectors.

# Fields
$(TYPEDFIELDS)
"""
struct OrderedDistribution{D<:ContinuousMultivariateDistribution,B} <:
       ContinuousMultivariateDistribution
    "distribution transformed to have ordered support"
    dist::D
    "transformation from constrained space to ordered unconstrained space"
    transform::B
end

"""
    ordered(d::Distribution)

Return a `Distribution` whose support are ordered vectors, i.e., vectors with increasingly ordered elements.

Specifically, `d` is restricted to the subspace of its domain containing only ordered elements.

!!! warning
    `rand` is implemented using rejection sampling, which can be slow for high-dimensional distributions.
    In such cases, consider using MCMC methods to sample from the distribution instead.

!!! warning
    The resulting ordered distribution is un-normalized, which can cause issues in some contexts, e.g. in
    hierarchical models where the parameters of the ordered distribution are themselves sampled.
    See the notes below for a more detailed discussion.

## Notes on `ordered` being un-normalized

The resulting ordered distribution is un-normalized. This is not a problem if used in a context where the
normalizing factor is irrelevant, but if the value of the normalizing factor impacts the resulting computation,
the results may be inaccurate.

For example, if the distribution is used in sampling a posterior distribution with MCMC and the parameters
of the ordered distribution are themselves sampled, then the normalizing factor would in general be needed
for accurate sampling, and `ordered` should not be used. However, if the parameters are fixed, then since
MCMC does not require distributions be normalized, `ordered` may be used without problems.

A common case is where the distribution being ordered is a joint distribution of `n` identical univariate
distributions. In this case the normalization factor works out to be the constant `n!`, and `ordered` can
again be used without problems even if the parameters of the univariate distribution are sampled.
"""
function ordered(d::ContinuousMultivariateDistribution)
    # We're good if the map from unconstrained (in which we apply the ordered bijector)
    # to constrained is monotonically increasing, i.e. order-preserving. In that case,
    # we can form the ordered transformation as `binv ∘ OrderedBijector() ∘ b`.
    # Similarly, if we're working with monotonically decreasing maps, we can do the same
    # but with the addition of a sign flip before and after the ordered bijector.
    b = bijector(d)
    binv = inverse(b)
    ordered_b = if is_monotonically_decreasing(binv)
        SignFlip() ∘ inverse(OrderedBijector()) ∘ SignFlip() ∘ b
    elseif is_monotonically_increasing(binv)
        inverse(OrderedBijector()) ∘ b
    else
        throw(ArgumentError("ordered transform is currently not supported for $d."))
    end

    return OrderedDistribution(d, ordered_b)
end

bijector(d::OrderedDistribution) = d.transform

Base.eltype(::Type{<:OrderedDistribution{D}}) where {D} = eltype(D)
Base.eltype(d::OrderedDistribution) = eltype(d.dist)
function Distributions._logpdf(d::OrderedDistribution, x::AbstractVector{<:Real})
    lp = Distributions.logpdf(d.dist, x)
    issorted(x) && return lp
    return oftype(lp, -Inf)
end
Base.length(d::OrderedDistribution) = length(d.dist)

function Distributions._rand!(
    rng::AbstractRNG, d::OrderedDistribution, x::AbstractVector{<:Real}
)
    # Rejection sampling.
    while true
        Distributions.rand!(rng, d.dist, x)
        issorted(x) && return x
    end
end
