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
