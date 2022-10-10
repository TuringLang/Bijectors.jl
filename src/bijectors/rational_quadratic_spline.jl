"""
    RationalQuadraticSpline{T} <: Bijector

Implementation of the Rational Quadratic Spline flow [1].

- Outside of the interval `[minimum(widths), maximum(widths)]`, this mapping is given 
  by the identity map. 
- Inside the interval it's given by a monotonic spline (i.e. monotonic polynomials 
  connected at intermediate points) with endpoints fixed so as to continuously transform
  into the identity map.

For the sake of efficiency, there are separate implementations for 0-dimensional and
1-dimensional inputs.

# Notes
There are two constructors for `RationalQuadraticSpline`:
- `RationalQuadraticSpline(widths, heights, derivatives)`: it is assumed that `widths`, 
`heights`, and `derivatives` satisfy the constraints that makes this a valid bijector, i.e.
  - `widths`: monotonically increasing and `length(widths) == K`,
  - `heights`: monotonically increasing and `length(heights) == K`,
  - `derivatives`: non-negative and `derivatives[1] == derivatives[end] == 1`.
- `RationalQuadraticSpline(widths, heights, derivatives, B)`: other than than the lengths, 
    no assumptions are made on parameters. Therefore we will transform the parameters s.t.:
  - `widths_new` ∈ [-B, B]ᴷ⁺¹, where `K == length(widths)`,
  - `heights_new` ∈ [-B, B]ᴷ⁺¹, where `K == length(heights)`,
  - `derivatives_new` ∈ (0, ∞)ᴷ⁺¹ with `derivatives_new[1] == derivates_new[end] == 1`, 
    where `(K - 1) == length(derivatives)`.

# Examples
## Univariate
```jldoctest
julia> using StableRNGs: StableRNG; rng = StableRNG(42);  # For reproducibility.

julia> using Bijectors: RationalQuadraticSpline

julia> K = 3; B = 2;

julia> # Monotonic spline on '[-B, B]' with `K` intermediate knots/"connection points".
       b = RationalQuadraticSpline(randn(rng, K), randn(rng, K), randn(rng, K - 1), B);

julia> b(0.5) # inside of `[-B, B]` → transformed
1.1943325397834206

julia> b(5.) # outside of `[-B, B]` → not transformed
5.0

julia> b = RationalQuadraticSpline(b.widths, b.heights, b.derivatives);

julia> b(0.5) # inside of `[-B, B]` → transformed
1.1943325397834206

julia> d = 2; K = 3; B = 2;

julia> b = RationalQuadraticSpline(randn(rng, d, K), randn(rng, d, K), randn(rng, d, K - 1), B);

julia> b([-1., 1.])
2-element Vector{Float64}:
 -1.5660106244288925
  0.5384702734738573

julia> b([-5., 5.])
2-element Vector{Float64}:
 -5.0
  5.0

julia> b([-1., 5.])
2-element Vector{Float64}:
 -1.5660106244288925
  5.0
```

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
struct RationalQuadraticSpline{T} <: Bijector
    widths::T      # K widths
    heights::T     # K heights
    derivatives::T # K derivatives, with endpoints being ones

    function RationalQuadraticSpline(
        widths::T,
        heights::T,
        derivatives::T
    ) where {T<:AbstractVector}
        # TODO: add a `NoArgCheck` type and argument so we can circumvent if we want        
        @assert length(widths) == length(heights) == length(derivatives)
        @assert all(derivatives .> 0) "derivatives need to be positive"
        
        return new{T}(widths, heights, derivatives)
    end

    function RationalQuadraticSpline(
        widths::T,
        heights::T,
        derivatives::T
    ) where {T<:AbstractMatrix}
        @assert size(widths, 2) == size(heights, 2) == size(derivatives, 2)
        @assert all(derivatives .> 0) "derivatives need to be positive"
        return new{T}(widths, heights, derivatives)
    end
end

function RationalQuadraticSpline(
    widths::A,
    heights::A, 
    derivatives::A,
    B::T2
) where {T1, T2, A <: AbstractVector{T1}}
    return RationalQuadraticSpline(
        (cumsum(vcat([zero(T1)], LogExpFunctions.softmax(widths))) .- 0.5) * 2 * B,
        (cumsum(vcat([zero(T1)], LogExpFunctions.softmax(heights))) .- 0.5) * 2 * B,
        vcat([one(T1)], LogExpFunctions.log1pexp.(derivatives), [one(T1)])
    )
end

function RationalQuadraticSpline(
    widths::A,
    heights::A,
    derivatives::A,
    B::T2
) where {T1, T2, A <: AbstractMatrix{T1}}
    ws = hcat(zeros(T1, size(widths, 1)), LogExpFunctions.softmax(widths; dims = 2))
    hs = hcat(zeros(T1, size(widths, 1)), LogExpFunctions.softmax(heights; dims = 2))
    ds = hcat(ones(T1, size(widths, 1)), LogExpFunctions.log1pexp.(derivatives), ones(T1, size(widths, 1)))

    return RationalQuadraticSpline(
        (2 * B) .* (cumsum(ws; dims = 2) .- 0.5),
        (2 * B) .* (cumsum(hs; dims = 2) .- 0.5),
        ds
    )
end

##########################
### Forward evaluation ###
##########################
function rqs_univariate(widths, heights, derivatives, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    # We're working on [-B, B] and `widths[end]` is `B`
    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return one(T) * x
    end

    K = length(widths)

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst(widths, x) - 1

    # Width
    # If k == 0 then we should put it in the bin `[-B, widths[1]]`
    w_k = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - w_k

    # Slope
    h_k = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - h_k

    s = Δy / w
    ξ = (x - w_k) / w

    # Derivatives at knot-points
    # Note that we have (K - 1) knot-points, not K
    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Eq. (14)
    numerator = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    denominator = s + (d_kplus1 + d_k - 2s) * ξ * (1 - ξ)
    g = h_k + numerator / denominator

    return g
end


# univariate
function transform(b::RationalQuadraticSpline{<:AbstractVector}, x::Real)
    return rqs_univariate(b.widths, b.heights, b.derivatives, x)
end

# multivariate
# TODO: Improve.
function transform(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVector)
    return [rqs_univariate(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], x[i]) for i = 1:length(x)]
end

##########################
### Inverse evaluation ###
##########################
function rqs_univariate_inverse(widths, heights, derivatives, y::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(y))

    if (y ≤ -heights[end]) || (y ≥ heights[end])
        return one(T) * y
    end

    K = length(widths)
    k = searchsortedfirst(heights, y) - 1

    # Width
    w_k = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - w_k

    # Slope
    h_k = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - h_k

    # Recurring quantities
    s = Δy / w
    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]
    ds = d_kplus1 + d_k - 2 * s

    # Eq. (25)
    a1 = Δy * (s - d_k) + (y - h_k) * ds
    # Eq. (26)
    a2 = Δy * d_k - (y - h_k) * ds
    # Eq. (27)
    a3 = - s * (y - h_k)

    # Eq. (24). There's a mistake in the paper; says `x` but should be `ξ`
    numerator = - 2 * a3
    denominator = (a2 + sqrt(a2^2 - 4 * a1 * a3))
    ξ = numerator / denominator

    return ξ * w + w_k
end

function transform(ib::Inverse{<:RationalQuadraticSpline}, y::Real)
    return rqs_univariate_inverse(ib.orig.widths, ib.orig.heights, ib.orig.derivatives, y)
end

# TODO: Improve.
function transform(ib::Inverse{<:RationalQuadraticSpline}, y::AbstractVector)
    b = ib.orig
    return [rqs_univariate_inverse(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], y[i]) for i = 1:length(y)]
end

######################
### `logabsdetjac` ###
######################
function rqs_logabsdetjac(widths, heights, derivatives, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(y))
    K = length(widths) - 1
    
    # Find which bin `x` is in
    k = searchsortedfirst(widths, x) - 1

    if k > K || k == 0
        return zero(T) * x
    end

    # Width
    w = widths[k + 1] - widths[k]

    # Slope
    Δy = heights[k + 1] - heights[k]

    # Recurring quantities
    s = Δy / w
    ξ = (x - widths[k]) / w

    numerator = s^2 * (derivatives[k + 1] * ξ^2
                       + 2 * s * ξ * (1 - ξ)
                       + derivatives[k] * (1 - ξ)^2)
    denominator = s + (derivatives[k + 1] + derivatives[k] - 2 * s) * ξ * (1 - ξ)

    return log(numerator) - 2 * log(denominator)
end

function rqs_logabsdetjac(
    widths::AbstractVector,
    heights::AbstractVector,
    derivatives::AbstractVector,
    x::Real
)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return zero(T) * x
    end

    K = length(widths)
    k = searchsortedfirst(widths, x) - 1

    # Width
    w_k = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - w_k

    # Slope
    h_k = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - h_k

    # Recurring quantities
    s = Δy / w
    ξ = (x - w_k) / w

    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    numerator = s^2 * (d_kplus1 * ξ^2 + 2 * s * ξ * (1 - ξ) + d_k * (1 - ξ)^2)
    denominator = s + (d_kplus1 + d_k - 2 * s) * ξ * (1 - ξ)

    return log(numerator) - 2 * log(denominator)
end

function logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector}, x::Real)
    return rqs_logabsdetjac(b.widths, b.heights, b.derivatives, x)
end

# TODO: Improve.
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVector)
    return sum([
        rqs_logabsdetjac(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], x[i])
        for i = 1:length(x)
    ])
end

#################
### `forward` ###
#################

# TODO: implement this for `x::AbstractVector` and similarily for 1-dimensional `b`,
# and possibly inverses too?
function rqs_forward(
    widths::AbstractVector,
    heights::AbstractVector,
    derivatives::AbstractVector,
    x::Real
)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return (one(T) * x, zero(T) * x)
    end

    # Find which bin `x` is in
    K = length(widths)
    k = searchsortedfirst(widths, x) - 1

    # Width
    w_k = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - w_k

    # Slope
    h_k = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - h_k

    # Recurring quantities
    s = Δy / w
    ξ = (x - w_k) / w

    d_k = (k == 0) ? one(T) : derivatives[k]
    d_kplus1 = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Re-used for both `logjac` and `y`
    denominator = s + (d_kplus1 + d_k - 2 * s) * ξ * (1 - ξ)

    # logjac
    numerator_jl = s^2 * (d_kplus1 * ξ^2 + 2 * s * ξ * (1 - ξ) + d_k * (1 - ξ)^2)
    logjac = log(numerator_jl) - 2 * log(denominator)

    # y
    numerator_y = Δy * (s * ξ^2 + d_k * ξ * (1 - ξ))
    y = h_k + numerator_y / denominator

    return (y, logjac)
end

function with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractVector}, x::Real)
    return rqs_forward(b.widths, b.heights, b.derivatives, x)
end

function with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVector)
    return transform(b, x), logabsdetjac(b, x)
end
