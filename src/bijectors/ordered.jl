struct SignFlip <: Bijector end

with_logabsdet_jacobian(::SignFlip, x) = -x, zero(eltype(x))
inverse(::SignFlip) = SignFlip()
output_size(::SignFlip, dim) = dim
is_monotonically_increasing(::SignFlip) = false
is_monotonically_decreasing(::SignFlip) = true

abstract type AbstractOrdering end
struct Ascending <: AbstractOrdering end
struct Descending <: AbstractOrdering end
struct FixedOrder{ordertuple} <: AbstractOrdering end

"""
    OrderedBijector()

A bijector mapping unordered vectors in ℝᵈ to ordered vectors in ℝᵈ.

## See also
- [Stan's documentation](https://mc-stan.org/docs/2_27/reference-manual/ordered-vector.html)
  - Note that this transformation and its inverse are the _opposite_ of in this reference.
"""
struct OrderedBijector{OT<:AbstractOrdering} <: Bijector
    OrderedBijector() = new{Ascending}()
    OrderedBijector(::OT) where {OT <: AbstractOrdering}= new{OT}()
    OrderedBijector{OT}() where {OT <: AbstractOrdering} = new{OT}()
    OrderedBijector(ordertuple::Tuple{Int, Int, Vararg{Int}}) = OrderedBijector{FixedOrder{ordertuple}}()
    function OrderedBijector{FixedOrder{ordertuple}}() where {ordertuple}
        @assert (ordertuple isa Tuple{Int, Int, Vararg{Int}}) && all(ordertuple .> 0) && allunique(ordertuple)
        new{FixedOrder{ordertuple}}()
    end
end

with_logabsdet_jacobian(b::OrderedBijector, y) = transform(b, y), logabsdetjac(b, y)

transform(b::OrderedBijector{OT}, y::AbstractVecOrMat) where {OT<:AbstractOrdering} = _transform_ordered(y, OT)

function _transform_ordered(y::AbstractVector, ::Type{Ascending})
    x = similar(y)
    @assert !isempty(y)

    @inbounds x[1] = y[1]
    @inbounds for i in 2:length(x)
        x[i] = x[i - 1] + exp(y[i])
    end

    return x
end

function _transform_ordered(y::AbstractMatrix, ::Type{Ascending})
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

function _transform_ordered(y::AbstractVector, ::Type{Descending})
    x = similar(y)
    @assert !isempty(y)

    N = length(x)
    @inbounds x[N] = y[N]
    @inbounds for i in N-1:-1:1
        x[i] = x[i + 1] + exp(y[i])
    end

    return x
end

function _transform_ordered(y::AbstractMatrix, ::Type{Descending})
    x = similar(y)
    @assert !isempty(y)

    N = size(x, 1)
    @inbounds for j in 1:size(x, 2), i in N:-1:1
        if i == N
            x[i, j] = y[i, j]
        else
            x[i, j] = x[i + 1, j] + exp(y[i, j])
        end
    end

    return x
end


function _transform_ordered(y::AbstractVector, ::Type{FixedOrder{o}}) where {o}

    x = copy(y)
    @assert !isempty(y)

    @inbounds x[o[1]] = y[o[1]]
    @inbounds for i in 2:length(o)
        x[o[i]] = x[o[i - 1]] + exp(y[o[i]])
    end

    return x
end

function _transform_ordered(y::AbstractMatrix, ::Type{FixedOrder{o}}) where {o}
    x = copy(y)
    @assert !isempty(y)

    @inbounds for j in 1:size(x, 2), i in 1:length(o)
        if i == 1
            x[o[i], j] = y[o[i], j]
        else
            x[o[i], j] = x[o[i - 1], j] + exp(y[o[i], j])
        end
    end

    return x
end

transform(ib::Inverse{OrderedBijector{OT}}, x::AbstractVecOrMat) where {OT <: AbstractOrdering} = _transform_inverse_ordered(x, OT)

function _transform_inverse_ordered(x::AbstractVector, ::Type{Ascending})
    y = similar(x)
    @assert !isempty(y)

    @inbounds y[1] = x[1]
    @inbounds for i in 2:length(x)
        @inbounds y[i] = log(x[i] - x[i - 1])
    end

    return y
end

function _transform_inverse_ordered(x::AbstractMatrix, ::Type{Ascending})
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

function _transform_inverse_ordered(x::AbstractVector, ::Type{Descending})
    y = similar(x)
    @assert !isempty(y)

    n = length(x)
    @inbounds y[n] = x[n]
    @inbounds for i in (n - 1):-1:1
        @inbounds y[i] = log(x[i] - x[i + 1])
    end

    return y
end

function _transform_inverse_ordered(x::AbstractMatrix, ::Type{Descending})
    y = similar(x)
    @assert !isempty(y)
    n = size(y, 1)
    @inbounds for j in 1:size(y, 2), i in n:-1:1
        if i == n
            y[i, j] = x[i, j]
        else
            y[i, j] = log(x[i, j] - x[i + 1, j])
        end
    end

    return y
end

function _transform_inverse_ordered(x::AbstractVector, ::Type{FixedOrder{o}}) where {o}
    y = copy(x)
    @assert !isempty(y)

    @inbounds y[o[1]] = x[o[1]]
    @inbounds for i in 2:length(o)
        @inbounds y[o[i]] = log(x[o[i]] - x[o[i - 1]])
    end

    return y
end

function _transform_inverse_ordered(x::AbstractMatrix, ::Type{FixedOrder{o}}) where {o}
    y = copy(x)
    @assert !isempty(y)

    @inbounds for j in 1:size(y, 2), i in 1:length(o)
        if i == 1
            y[o[i], j] = x[o[i], j]
        else
            y[o[i], j] = log(x[o[i], j] - x[o[i - 1], j])
        end
    end

    return y
end

logabsdetjac(::OrderedBijector{Ascending}, x::AbstractVector) = sum(@view(x[2:end]))
logabsdetjac(::OrderedBijector{Ascending}, x::AbstractMatrix) = vec(sum(@view(x[2:end, :]); dims=1))

logabsdetjac(::OrderedBijector{Descending}, x::AbstractVector) = sum(@view(x[1:end-1]))
logabsdetjac(::OrderedBijector{Descending}, x::AbstractMatrix) = vec(sum(@view(x[1:end-1, :]); dims=1))

logabsdetjac(::OrderedBijector{FixedOrder{o}}, x::AbstractVector) where {o} = sum(@view(x[collect(o)[2:end]]))
logabsdetjac(::OrderedBijector{FixedOrder{o}}, x::AbstractMatrix) where {o} = vec(sum(@view(x[collect(o)[2:end], :]); dims=1))

# Need a custom distribution type to handle this properly.
"""
    OrderedDistribution

Wraps a distribution to restrict its support to the subspace of ordered vectors.

# Fields
$(TYPEDFIELDS)
"""
struct OrderedDistribution{D<:ContinuousMultivariateDistribution, B, OT<:AbstractOrdering} <:
       ContinuousMultivariateDistribution
    "distribution transformed to have ordered support"
    dist::D
    "transformation from constrained space to ordered unconstrained space"
    transform::B

    OrderedDistribution(d::D, b::B, ::OT=Ascending()) where {D, B, OT <: AbstractOrdering} = new{D, B, OT}(d, b)
    OrderedDistribution{D, B}(d::D, b::B, ::OT=Ascending()) where {D, B, OT <: AbstractOrdering} = new{D, B, OT}(d, b)
    OrderedDistribution(d::D, b::B, ordertuple::Tuple{Int, Int, Vararg{Int}}) where {D, B} = OrderedDistribution(d, b, FixedOrder{ordertuple})
    function OrderedDistribution(d::D, b::B, ::FixedOrder{ordertuple}) where {D, B, ordertuple}
        if !((ordertuple isa Tuple{Int, Int, Vararg{Int}})
             && issubset(ordertuple, 1:length(d))
             && allunique(ordertuple))
        throw(ArgumentError("ordertuple must be a subset of 1:$(length(d)) of length at least 2 with no duplicates."))
        end
        new{D, B, FixedOrder{ordertuple}}(d, b)
    end
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

ordered(d::ContinuousMultivariateDistribution, ordertuple::Tuple{Int, Int, Vararg{Int}}) = ordered(d, FixedOrder{ordertuple}())

function ordered(d::ContinuousMultivariateDistribution, order::OT=Ascending()) where {OT <: AbstractOrdering}
    # We're good if the map from unconstrained (in which we apply the ordered bijector)
    # to constrained is monotonically increasing, i.e. order-preserving. In that case,
    # we can form the ordered transformation as `binv ∘ OrderedBijector() ∘ b`.
    # Similarly, if we're working with monotonically decreasing maps, we can do the same
    # but with the addition of a sign flip before and after the ordered bijector.
    b = bijector(d)
    binv = inverse(b)
    ordered_b = if is_monotonically_decreasing(binv)
        SignFlip() ∘ inverse(OrderedBijector{OT}()) ∘ SignFlip() ∘ b
    elseif is_monotonically_increasing(binv)
        inverse(OrderedBijector{OT}()) ∘ b
    else
        throw(ArgumentError("ordered transform is currently not supported for $d."))
    end

    return OrderedDistribution(d, ordered_b, order)
end

bijector(d::OrderedDistribution) = d.transform

Base.eltype(::Type{<:OrderedDistribution{D}}) where {D} = eltype(D)
Base.eltype(d::OrderedDistribution) = eltype(d.dist)
function Distributions._logpdf(d::OrderedDistribution{D, B, OT}, x::AbstractVector{<:Real}) where {D, B, OT}
    lp = Distributions.logpdf(d.dist, x)
    return if _is_ordered(x, OT)
        lp
    else
        oftype(lp, -Inf)
    end
end
Base.length(d::OrderedDistribution) = length(d.dist)

function Distributions._rand!(
    rng::AbstractRNG, d::OrderedDistribution{D, B, OT}, x::AbstractVector{<:Real}
) where {D, B, OT}
    # Rejection sampling.
    while true
        Distributions.rand!(rng, d.dist, x)
        _is_ordered(x, OT) && return x
    end
    return x
end

_is_ordered(x::AbstractVector, ::Type{Ascending}) = issorted(x)
_is_ordered(x::AbstractVector, ::Type{Descending}) = issorted(x, rev=true)

function _is_ordered(x::AbstractVector, ::Type{FixedOrder{o}}) where {o}
    @inbounds for i in 2:length(o)
        x[o[i-1]] ≤ x[o[i]] || return false
    end
    true
end