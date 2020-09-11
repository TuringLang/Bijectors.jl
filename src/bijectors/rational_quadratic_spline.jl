using NNlib

"""
    RationalQuadraticSpline{T, 0} <: Bijector{0}
    RationalQuadraticSpline{T, 1} <: Bijector{1}

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
- `RationalQuadraticSpline(widths, heights, derivatives)`: it is assumed that `widths`, `heights`,
  and `derivatives` satisfy the constraints that makes this a valid bijector, i.e.
  - `widths`: monotonically increasing and `length(widths) == K`,
  - `heights`: monotonically increasing and `length(heights) == K`,
  - `derivatives`: non-negative and `derivatives[1] == derivatives[end] == 1`.
- `RationalQuadraticSpline(widths, heights, derivatives, B)`: other than than the lengths, no 
  assumptions are made on parameters. Therefore we will transform the parameters so that:
  - `widths_new` ∈ [-B, B]ᴷ⁺¹, where `K == length(widths)`,
  - `heights_new` ∈ [-B, B]ᴷ⁺¹, where `K == length(heights)`,
  - `derivatives_new` ∈ (0, ∞)ᴷ⁺¹ with `derivatives_new[1] == derivates_new[end] == 1`, where 
    `(K - 1) == length(derivatives)`.

# Examples
## Univariate
```julia-repl
julia> using Bijectors: RationalQuadraticSpline

julia> K = 3; B = 2;

julia> # Monotonic spline on '[-B, B]' with `K` intermediate knots/"connection points".
       b = RationalQuadraticSpline(randn(K), randn(K), randn(K - 1), B);

julia> b(0.5) # inside of `[-B, B]` → transformed
1.412300607463467

julia> b(5.) # outside of `[-B, B]` → not transformed
5.0
```
Or we can use the constructor with the parameters correctly constrained:
```julia-repl
julia> b = RationalQuadraticSpline(b.widths, b.heights, b.derivatives);

julia> b(0.5) # inside of `[-B, B]` → transformed
1.412300607463467
```
## Multivariate
```julia-repl
julia> d = 2; K = 3; B = 2;

julia> b = RationalQuadraticSpline(randn(d, K), randn(d, K), randn(d, K - 1), B);

julia> b([-1., 1.])
2-element Array{Float64,1}:
 -1.2568224171342797
  0.5537259740554675

julia> b([-5., 5.])
2-element Array{Float64,1}:
 -5.0
  5.0

julia> b([-1., 5.])
2-element Array{Float64,1}:
 -1.2568224171342797
  5.0
```

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
struct RationalQuadraticSpline{T, N} <: Bijector{N}
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
        
        return new{T, 0}(widths, heights, derivatives)
    end

    function RationalQuadraticSpline(
        widths::T,
        heights::T,
        derivatives::T
    ) where {T<:AbstractMatrix}
        @assert size(widths, 2) == size(heights, 2) == size(derivatives, 2)
        @assert all(derivatives .> 0) "derivatives need to be positive"
        return new{T, 1}(widths, heights, derivatives)
    end
end

function RationalQuadraticSpline(
    widths::A,
    heights::A, 
    derivatives::A,
    B::T2
) where {T1, T2, A <: AbstractVector{T1}}
    # Using `NNLlinb.softax` instead of `StatsFuns.softmax` (which does inplace operations)
    return RationalQuadraticSpline(
        (cumsum(vcat([zero(T1)], NNlib.softmax(widths))) .- 0.5) * 2 * B,
        (cumsum(vcat([zero(T1)], NNlib.softmax(heights))) .- 0.5) * 2 * B,
        vcat([one(T1)], softplus.(derivatives), [one(T1)])
    )
end

function RationalQuadraticSpline(
    widths::A,
    heights::A,
    derivatives::A,
    B::T2
) where {T1, T2, A <: AbstractMatrix{T1}}
    ws = hcat(zeros(T1, size(widths, 1)), NNlib.softmax(widths; dims = 2))
    hs = hcat(zeros(T1, size(widths, 1)), NNlib.softmax(heights; dims = 2))
    ds = hcat(ones(T1, size(widths, 1)), softplus.(derivatives), ones(T1, size(widths, 1)))

    return RationalQuadraticSpline(
        (2 * B) .* (cumsum(ws; dims = 2) .- 0.5),
        (2 * B) .* (cumsum(hs; dims = 2) .- 0.5),
        ds
    )
end

function rqs_univariate(widths, heights, derivatives, x::Real)
    K = length(widths) - 1

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst(widths, x) - 1
    # If less than the smallest element or greater than largest element, use identity map
    if k > K || k == 0
        return x
    end

    # Width
    w = widths[k + 1] - widths[k]

    # Slope
    Δy = heights[k + 1] - heights[k]

    s = Δy / w
    ξ = (x - widths[k]) / w

    # Eq. (14)
    numerator = Δy * (s * ξ^2 + derivatives[k] * ξ * (1 - ξ))
    denominator = s + (derivatives[k + 1] + derivatives[k] - 2s) * ξ * (1 - ξ)
    g = heights[k] + numerator / denominator

    return g
end
# univariate
function (b::RationalQuadraticSpline{<:AbstractVector, 0})(x::Real)
    return rqs_univariate(b.widths, b.heights, b.derivatives, x)
end
(b::RationalQuadraticSpline{<:AbstractVector, 0})(x::AbstractVector) = b.(x)

# multivariate
function (b::RationalQuadraticSpline{<:AbstractMatrix, 1})(x::AbstractVector)
    @assert length(x) == size(b.widths, 1) == size(b.heights, 1) == size(b.derivatives, 1)
    
    return [rqs_univariate(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], x[i])
            for i = 1:length(x)]
end
function (b::RationalQuadraticSpline{<:AbstractMatrix, 1})(x::AbstractMatrix)
    return foldl(hcat, [b(x[:, i]) for i = 1:size(x, 2)])
end


function rqs_univariate_inverse(widths, heights, derivatives, y::Real)
    K = length(widths) - 1
    k = searchsortedfirst(heights, y) - 1

    if k > K || k == 0
        return y
    end

    # Width
    w = widths[k + 1] - widths[k]

    # Slope
    Δy = heights[k + 1] - heights[k]

    # Recurring quantities
    s = Δy / w
    ds = derivatives[k + 1] + derivatives[k] - 2 * s

    # Eq. (25)
    a1 = Δy * (s - derivatives[k]) + (y - heights[k]) * ds
    # Eq. (26)
    a2 = Δy * derivatives[k] - (y - heights[k]) * ds
    # Eq. (27)
    a3 = - s * (y - heights[k])

    # Eq. (24). There's a mistake in the paper; says `x` but should be `ξ`
    numerator = - 2 * a3
    denominator = (a2 + sqrt(a2^2 - 4 * a1 * a3))
    ξ = numerator / denominator
    
    return ξ * w + widths[k]
end
function (ib::Inverse{<:RationalQuadraticSpline, 0})(y::Real)
    return rqs_univariate_inverse(ib.orig.widths, ib.orig.heights, ib.orig.derivatives, y)
end
(ib::Inverse{<:RationalQuadraticSpline, 0})(y::AbstractVector) = ib.(y)

function (ib::Inverse{<:RationalQuadraticSpline, 1})(y::AbstractVector)
    b = ib.orig
    @assert length(y) == size(b.widths, 2) == size(b.heights, 2) == size(b.derivatives, 2)

    return [
        rqs_univariate_inverse(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], y[i])
        for i = 1:length(y)
    ]
end
function (ib::Inverse{<:RationalQuadraticSpline, 1})(y::AbstractMatrix)
    return mapreduce(ib, hcat, eachcol(y))
    # return foldl(hcat, [ib(y[:, i]) for i = 1:size(y, 2)])
end

function rqs_logabsdetjac(widths, heights, derivatives, x::Real)
    K = length(widths) - 1
    
    # Find which bin `x` is in
    k = searchsortedfirst(widths, x) - 1

    if k > K || k == 0
        return zero(eltype(x))
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
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::Real)
    return rqs_logabsdetjac(b.widths, b.heights, b.derivatives, x)
end
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractVector)
    return logabsdetjac.(b, x)
end
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractVector)
    return sum([
        rqs_logabsdetjac(b.widths[i, :], b.heights[i, :], b.derivatives[i, :], x[i])
        for i = 1:length(x)
    ])
end
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractMatrix)
    return map(x -> logabsdetjac(b, x), eachcol(x))
end

# TODO: implement this for `x::AbstractVector` and similarily for 1-dimensional `b`
function forward(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::Real)
    K = length(b.widths) - 1
    
    # Find which bin `x` is in
    k = searchsortedfirst(b.widths, x) - 1
    if k > K || k == 0
        return (rv = x, logabsdetjac = zero(eltype(x)))
    end

    # Width
    w = b.widths[k + 1] - b.widths[k]

    # Slope
    Δy = b.heights[k + 1] - b.heights[k]

    # Recurring quantities
    s = Δy / w
    ξ = (x - b.widths[k]) / w

    # Re-used for both `logjac` and `y`
    denominator = s + (b.derivatives[k + 1] + b.derivatives[k] - 2 * s) * ξ * (1 - ξ)

    # logjac
    numerator_jl = s^2 * (b.derivatives[k + 1] * ξ^2
                          + 2 * s * ξ * (1 - ξ)
                          + b.derivatives[k] * (1 - ξ)^2)
    logjac = log(numerator_jl) - 2 * log(denominator)

    # y
    numerator_y = Δy * (s * ξ^2 + b.derivatives[k] * ξ * (1 - ξ))
    y = b.heights[k] + numerator_y / denominator

    return (rv = y, logabsdetjac = logjac)
end
