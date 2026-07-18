#############################################
### Batched rational quadratic spline (RQS) ###
#############################################

# A batched counterpart to `RationalQuadraticSpline` that evaluates many splines over a
# batch of samples with whole-array operations, so the same source runs on `Array` and
# `CuArray` and is differentiable by every AD backend without hand-written rules.
#
# Parameter arrays carry the knot axis first: `(K + 1, D, N)` for `K` bins, `D` transformed
# dimensions, and `N` samples. Inputs are `(D, N)`.

# Constrain raw parameters into a monotone knot grid on `[-B, B]`, batched along dim 1.
# Mirrors the single-sample `RationalQuadraticSpline(widths, heights, derivatives, B)`
# constructor: softmax to positive increments, cumulative sum to knots, scale to `[-B, B]`.
function _rqs_constrain_knots(raw::AbstractArray, B)
    T = eltype(raw)
    Bc = T(B)
    increments = LogExpFunctions.softmax(raw; dims=1)
    lead = fill!(similar(raw, 1, Base.tail(size(raw))...), zero(T))
    return cumsum(cat(lead, increments; dims=1); dims=1) .* (2 * Bc) .- Bc
end

# Interior derivatives are made positive with softplus; the endpoints are fixed to one so
# the spline continues into the identity map outside `[-B, B]`.
function _rqs_constrain_derivatives(raw::AbstractArray)
    T = eltype(raw)
    edge = fill!(similar(raw, 1, Base.tail(size(raw))...), one(T))
    return cat(edge, LogExpFunctions.log1pexp.(raw), edge; dims=1)
end

"""
    rqs_params_from_raw(θ_raw::AbstractMatrix, n_dims::Integer, B)

Turn a matrix of raw neural-network outputs into constrained rational-quadratic-spline knot
parameters. `θ_raw` has shape `((3K - 1) * n_dims, N)`, laid out per dimension as `K` width
logits, `K` height logits, then `K - 1` derivative logits.

Returns `(widths, heights, derivatives)`, each `(K + 1, n_dims, N)`, with `widths` and
`heights` monotone on `[-B, B]` and `derivatives` positive with unit endpoints.
"""
function rqs_params_from_raw(θ_raw::AbstractMatrix, n_dims::Integer, B)
    n_params, N = size(θ_raw)
    K = (n_params ÷ n_dims + 1) ÷ 3
    θ = reshape(θ_raw, 3K - 1, n_dims, N)
    widths = _rqs_constrain_knots(θ[1:K, :, :], B)
    heights = _rqs_constrain_knots(θ[(K + 1):(2K), :, :], B)
    derivatives = _rqs_constrain_derivatives(θ[(2K + 1):(3K - 1), :, :])
    return widths, heights, derivatives
end

# Locate the bin of each element and whether it lies inside the spline range. `knots` is
# `(K + 1, D, N)`, `x` is `(D, N)`. `count` is the number of knots not exceeding `x`, so a
# point in `[knots[k], knots[k+1])` gives `count == k`; `count == 0` or `count == K + 1`
# means it is below or above the range. The comparison and integer reduction are
# non-differentiable by construction, which is what confines the gradient to the arithmetic
# of the selected bin.
function _rqs_bin(knots::AbstractArray, x::AbstractMatrix)
    K = size(knots, 1) - 1
    count = dropdims(sum(knots .<= reshape(x, 1, size(x)...); dims=1); dims=1)
    inside = (count .>= 1) .& (count .<= K)
    return clamp.(count, 1, K), inside
end

# Gather the lower and upper knot values of each element's bin into `(D, N)` arrays. Linear
# indices are built by broadcast so they live on the same device as the parameters, and the
# gather is a plain `getindex` by integer array: vectorized on the GPU and differentiable on
# every backend, with the gradient flowing back to the two selected knots.
function _rqs_gather(knots::AbstractArray, k::AbstractMatrix{<:Integer})
    stride1 = size(knots, 1)
    D, N = size(k)
    offset =
        reshape(0:(D - 1), D, 1) .* stride1 .+ reshape(0:(N - 1), 1, N) .* (stride1 * D)
    flat = reshape(knots, :)
    return flat[offset .+ k], flat[offset .+ (k .+ 1)]
end

"""
    rqs_forward(x, widths, heights, derivatives)

Evaluate the batched rational-quadratic spline forward. `x` is `(D, N)` and the knot
parameters are `(K + 1, D, N)`. Returns `(y, logjac)` with `y` of shape `(D, N)` and
`logjac` of shape `(1, N)`, the per-sample sum over dimensions of `log|dy/dx|`. Outside
`[widths[1], widths[end]]` the map is the identity and contributes zero to `logjac`.
"""
function rqs_forward(
    x::AbstractMatrix,
    widths::AbstractArray,
    heights::AbstractArray,
    derivatives::AbstractArray,
)
    T = eltype(x)
    k, inside = _rqs_bin(widths, x)
    xₖ, xₖ₊₁ = _rqs_gather(widths, k)
    yₖ, yₖ₊₁ = _rqs_gather(heights, k)
    dₖ, dₖ₊₁ = _rqs_gather(derivatives, k)

    Δx = xₖ₊₁ .- xₖ
    Δy = yₖ₊₁ .- yₖ
    s = Δy ./ Δx
    # Clamp keeps the discarded (out-of-range) branch finite so its zero-weighted gradient
    # never turns into NaN; inside the range ξ is already in [0, 1] and clamp is a no-op.
    ξ = clamp.((x .- xₖ) ./ Δx, zero(T), one(T))

    denom = @. s + (dₖ₊₁ + dₖ - 2s) * ξ * (1 - ξ)
    y_bin = @. yₖ + Δy * (s * ξ^2 + dₖ * ξ * (1 - ξ)) / denom

    y = ifelse.(inside, y_bin, x)
    logjac = ifelse.(inside, _rqs_forward_logjac(s, dₖ, dₖ₊₁, ξ), zero(T))
    return y, sum(logjac; dims=1)
end

# Forward log|dy/dx| for a bin at spline coordinate ξ in [0, 1], reused by the inverse.
function _rqs_forward_logjac(s, dₖ, dₖ₊₁, ξ)
    denom = @. s + (dₖ₊₁ + dₖ - 2s) * ξ * (1 - ξ)
    nom = @. dₖ₊₁ * ξ^2 + 2s * ξ * (1 - ξ) + dₖ * (1 - ξ)^2
    return @. 2 * log(abs(s)) + log(abs(nom)) - 2 * log(abs(denom))
end

"""
    rqs_inverse(y, widths, heights, derivatives)

Invert the batched rational-quadratic spline. `y` is `(D, N)` and the knot parameters are
`(K + 1, D, N)`. Returns `(x, logjac)` with `x` of shape `(D, N)` and `logjac` of shape
`(1, N)`, the per-sample sum over dimensions of `log|dx/dy|`. The bin holding each `y` is a
monotone quadratic in the spline coordinate, solved with the roundoff-stable root; the
log-det is the negation of the forward log-det at the recovered coordinate.
"""
function rqs_inverse(
    y::AbstractMatrix,
    widths::AbstractArray,
    heights::AbstractArray,
    derivatives::AbstractArray,
)
    T = eltype(y)
    k, inside = _rqs_bin(heights, y)
    xₖ, xₖ₊₁ = _rqs_gather(widths, k)
    yₖ, yₖ₊₁ = _rqs_gather(heights, k)
    dₖ, dₖ₊₁ = _rqs_gather(derivatives, k)

    Δx = xₖ₊₁ .- xₖ
    Δy = yₖ₊₁ .- yₖ
    s = Δy ./ Δx
    # Clamp to the bin so the discarded (out-of-range) branch stays finite; inside the range
    # Δy2 already lies in [0, Δy] and clamp is a no-op.
    Δy2 = clamp.(y .- yₖ, zero(T), Δy)

    c1 = dₖ₊₁ .+ dₖ .- 2 .* s
    a = @. Δy * (s - dₖ) + Δy2 * c1
    b = @. Δy * dₖ - Δy2 * c1
    c = @. -s * Δy2
    disc = @. max(b^2 - 4 * a * c, zero(T))
    denom = @. -b - sqrt(disc)
    ξ = clamp.((2 .* c) ./ denom, zero(T), one(T))

    x = ifelse.(inside, xₖ .+ ξ .* Δx, y)
    logjac = ifelse.(inside, .-_rqs_forward_logjac(s, dₖ, dₖ₊₁, ξ), zero(T))
    return x, sum(logjac; dims=1)
end

"""
    BatchedRQS(widths, heights, derivatives)
    BatchedRQS(θ_raw::AbstractMatrix, n_dims::Integer, B)

Batched rational quadratic spline bijector. The knot parameters `widths`, `heights`, and
`derivatives` are `(K + 1, D, N)` arrays already constrained to a valid monotone spline (as
produced by [`rqs_params_from_raw`](@ref)); the second constructor builds them from raw
neural-network outputs. `transform` maps `(D, N)` inputs to `(D, N)` outputs and
`logabsdetjac` returns the per-sample `(N,)` log-determinant.
"""
struct BatchedRQS{T<:AbstractArray} <: Bijector
    widths::T
    heights::T
    derivatives::T

    function BatchedRQS(widths::T, heights::T, derivatives::T) where {T<:AbstractArray}
        if !(size(widths) == size(heights) == size(derivatives))
            throw(
                DimensionMismatch("widths, heights, and derivatives must share their shape")
            )
        end
        return new{T}(widths, heights, derivatives)
    end
end

function BatchedRQS(θ_raw::AbstractMatrix, n_dims::Integer, B)
    return BatchedRQS(rqs_params_from_raw(θ_raw, n_dims, B)...)
end

function transform(b::BatchedRQS, x::AbstractMatrix)
    return first(rqs_forward(x, b.widths, b.heights, b.derivatives))
end

function transform(ib::Inverse{<:BatchedRQS}, y::AbstractMatrix)
    b = ib.orig
    return first(rqs_inverse(y, b.widths, b.heights, b.derivatives))
end

function logabsdetjac(b::BatchedRQS, x::AbstractMatrix)
    return vec(last(rqs_forward(x, b.widths, b.heights, b.derivatives)))
end

function with_logabsdet_jacobian(b::BatchedRQS, x::AbstractMatrix)
    y, logjac = rqs_forward(x, b.widths, b.heights, b.derivatives)
    return y, vec(logjac)
end

function with_logabsdet_jacobian(ib::Inverse{<:BatchedRQS}, y::AbstractMatrix)
    b = ib.orig
    x, logjac = rqs_inverse(y, b.widths, b.heights, b.derivatives)
    return x, vec(logjac)
end
