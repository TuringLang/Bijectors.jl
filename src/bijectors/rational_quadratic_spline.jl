using NNlib

struct RationalQuadraticSpline{T, D} <: Bijector{D}
    widths::T      # K widths
    heights::T     # K heights
    derivatives::T # (K - 1) derivatives

    function RationalQuadraticSpline(widths::T, heights::T, derivatives::T) where {T<:AbstractVector}
        # FIXME: add a `NoArgCheck` type and argument so we can circumvent if we want

        @assert length(widths) == length(heights) == length(derivatives) + 1 "widths $(length(widths)) ≠ heights $(length(heights)) ≠ $(length(derivatives)) + 1"
        # @assert all(widths .> 0) "widths need to be positive"
        # @assert all(heights .> 0) "heights need to be positive"
        # @assert widths[1] ≈ heights[1] "widths and heights need equal left endpoint"
        # @assert widths[end] ≈ heights[end] "widths and heights need equal right endpoint"
        # @assert sum(widths) ≈ sum(heights) "widths and heights should sum to 2B"  # should both sum to 2B s.t. [-B, B] × [-B, B]
        @assert all(derivatives .> 0) "derivatives need to be positive"

        return new{T, 0}(widths, heights, derivatives)
    end

    function RationalQuadraticSpline(widths::T, heights::T, derivatives::T) where {T<:AbstractMatrix}
        return new{T, 1}(widths, heights, derivatives)
    end
end

function RationalQuadraticSpline(widths::A, heights::A, derivatives::A, B::Real) where {A<:AbstractVecOrMat}
    return RationalQuadraticSpline(
        2 * B * (cumsum(NNlib.softmax(values(widths)); dims=1) .- 0.5),
        2 * B * (cumsum(NNlib.softmax(values(heights)); dims=1) .- 0.5),
        NNlib.softplus.(values(derivatives))
    )
end

function rqs_univariate(widths, heights, derivatives, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    # We're working on [-B, B] and `widths[end]` is `B`
    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return x
    end

    K = length(widths)

    # Find which bin `x` is in; subtract 1 because `searchsortedfirst` returns idx of ≥ not ≤
    k = searchsortedfirst(widths, x) - 1

    # Width
    # If k == 0 then we should put it in the bin `[-B, widths[1]]`
    wₖ = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - wₖ

    # Slope
    hₖ = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - hₖ

    s = Δy / w
    ξ = (x - wₖ) / w

    # Derivatives at knot-points
    # Note that we have (K - 1) knot-points, not K
    dₖ = (k == 0) ? one(T) : derivatives[k]
    dₖ₊₁ = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Eq. (14)
    numerator = Δy * (s * ξ^2 + dₖ * ξ * (1 - ξ))
    denominator = s + (dₖ₊₁ + dₖ - 2s) * ξ * (1 - ξ)
    g = hₖ + numerator / denominator

    # @info k w Δy s ξ numerator denominator

    return g
end

# univariate
(b::RationalQuadraticSpline{<:AbstractVector, 0})(x::Real) = rqs_univariate(b.widths, b.heights, b.derivatives, x)
(b::RationalQuadraticSpline{<:AbstractVector, 0})(x::AbstractVector) = b.(x)

# multivariate
function (b::RationalQuadraticSpline{<:AbstractMatrix, 1})(x::AbstractVector)
    @assert length(x) == size(b.widths, 2) == size(b.heights, 2) == size(b.derivatives, 2)

    return rqs_univariate.(eachcol(b.widths), eachcol(b.heights), eachcol(b.derivatives), x)
end
function (b::RationalQuadraticSpline{<:AbstractMatrix, 1})(x::AbstractMatrix)
    return foldl(hcat, [b(x[:, i]) for i = 1:size(x, 2)])
end

function rqs_univariate_inverse(widths, heights, derivatives, y::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(y))

    if (y ≤ -heights[end]) || (y ≥ heights[end])
        return y
    end

    K = length(widths)
    k = searchsortedfirst(heights, y) - 1

    # Width
    wₖ = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - wₖ

    # Slope
    hₖ = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - hₖ

    # Recurring quantities
    s = Δy / w
    dₖ = (k == 0) ? one(T) : derivatives[k]
    dₖ₊₁ = (k == K - 1) ? one(T) : derivatives[k + 1]
    ds = dₖ₊₁ + dₖ - 2 * s

    # Eq. (25)
    a1 = Δy * (s - dₖ) + (y - hₖ) * ds
    # Eq. (26)
    a2 = Δy * dₖ - (y - hₖ) * ds
    # Eq. (27)
    a3 = - s * (y - hₖ)

    # Eq. (24). There's a mistake in the paper; says `x` but should be `ξ`
    numerator = - 2 * a3
    denominator = (a2 + sqrt(a2^2 - 4 * a1 * a3))
    ξ = numerator / denominator

    # @assert isapprox(a1 * ξ^2 + a2 * ξ + a3, 0.0, atol=1e-8)

    return ξ * w + wₖ
end

(ib::Inversed{<:RationalQuadraticSpline, 0})(y::Real) = rqs_univariate_inverse(ib.orig.widths, ib.orig.heights, ib.orig.derivatives, y)
(ib::Inversed{<:RationalQuadraticSpline, 0})(y::AbstractVector) = ib.(y)

function (ib::Inversed{<:RationalQuadraticSpline, 1})(y::AbstractVector)
    b = ib.orig
    @assert length(y) == size(b.widths, 2) == size(b.heights, 2) == size(b.derivatives, 2)

    return [rqs_univariate_inverse(b.widths[:, i], b.heights[:, i], b.derivatives[:, i], y[i]) for i = 1:length(y)]
end
function (ib::Inversed{<:RationalQuadraticSpline, 1})(y::AbstractMatrix)
    return foldl(hcat, [ib(y[:, i]) for i = 1:size(y, 2)])
end

function rqs_logabsdetjac(widths::AbstractVector, heights::AbstractVector, derivatives::AbstractVector, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return zero(T)
    end

    K = length(widths)
    k = searchsortedfirst(widths, x) - 1

    # Width
    wₖ = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - wₖ

    # Slope
    hₖ = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - hₖ

    # Recurring quantities
    s = Δy / w
    ξ = (x - wₖ) / w

    dₖ = (k == 0) ? one(T) : derivatives[k]
    dₖ₊₁ = (k == K - 1) ? one(T) : derivatives[k + 1]

    numerator = s^2 * (dₖ₊₁ * ξ^2 + 2 * s * ξ * (1 - ξ) + dₖ * (1 - ξ)^2)
    denominator = s + (dₖ₊₁ + dₖ - 2 * s) * ξ * (1 - ξ)

    return log(numerator) - 2 * log(denominator)
end

logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::Real) = rqs_logabsdetjac(b.widths, b.heights, b.derivatives, x)
logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractVector) = logabsdetjac.(b, x)
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractVector)
    return sum(rqs_logabsdetjac.(eachcol(b.widths), eachcol(b.heights), eachcol(b.derivatives), x))
end
# TODO: improve this batch-impl
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractMatrix)
    return [logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)]
end

# TODO: implement this for `x::AbstractVector` and similarily for 1-dimensional `b`
function rqs_forward(widths::AbstractVector, heights::AbstractVector, derivatives::AbstractVector, x::Real)
    T = promote_type(eltype(widths), eltype(heights), eltype(derivatives), eltype(x))

    if (x ≤ -widths[end]) || (x ≥ widths[end])
        return (rv = x, logabsdetjac = zero(T))
    end

    # Find which bin `x` is in
    K = length(widths)
    k = searchsortedfirst(widths, x) - 1

    # Width
    wₖ = (k == 0) ? -widths[end] : widths[k]
    w = widths[k + 1] - wₖ

    # Slope
    hₖ = (k == 0) ? -heights[end] : heights[k]
    Δy = heights[k + 1] - hₖ

    # Recurring quantities
    s = Δy / w
    ξ = (x - wₖ) / w

    dₖ = (k == 0) ? one(T) : derivatives[k]
    dₖ₊₁ = (k == K - 1) ? one(T) : derivatives[k + 1]

    # Re-used for both `logjac` and `y`
    denominator = s + (dₖ₊₁ + dₖ - 2 * s) * ξ * (1 - ξ)

    # logjac
    numerator_jl = s^2 * (dₖ₊₁ * ξ^2 + 2 * s * ξ * (1 - ξ) + dₖ * (1 - ξ)^2)
    logjac = log(numerator_jl) - 2 * log(denominator)

    # y
    numerator_y = Δy * (s * ξ^2 + dₖ * ξ * (1 - ξ))
    y = hₖ + numerator_y / denominator

    return (rv = y, logabsdetjac = logjac)
end

function forward(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractVector)
    return rqs_forward(b.widths, b.heights, b.derivatives, x)
end

# TODO: this is probably overkill just go get the `forward` working in a batch-setting.
# Also probably veeeery slow for large arrays...
@generated function merge_nt(transforms::NamedTuple{names}, as::NamedTuple{names}...) where {names}
    exprs = []
    for k in names
        # push!(exprs, :($k = fs.$k((getfield.(as, $(QuoteNode(k))))...)))
        arr = Any[ :(getfield(as[$i], $(QuoteNode(k)))) for i in 1:length(as) ]
        push!(exprs, :($k = transforms.$k($(arr...))))
    end

    # return exprs
    return :(($(exprs...), ))
end

function rqs_forward(widths::AbstractVector, heights::AbstractVector, derivatives::AbstractVector, x::AbstractVector)
    return merge_nt((rv = vcat, logabsdetjac = vcat), (rqs_forward.(eachcol(widths), eachcol(heights), eachcol(derivatives), x))...)
end

function forward(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractMatrix)
    return rqs_forward(b.widths, b.heights, b.derivatives, x)
end

function rqs_forward(widths::AbstractMatrix, heights::AbstractMatrix, derivatives::AbstractMatrix, x::AbstractVector)
    return merge_nt((rv = vcat, logabsdetjac = +), (rqs_forward.(eachcol(widths), eachcol(heights), eachcol(derivatives), x))...)
end

function forward(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractMatrix)
    return merge_nt((rv = hcat, logabsdetjac = vcat), (rqs_forward.(Ref(b.widths), Ref(b.heights), Ref(b.derivatives), eachcol(x)))...)
end
