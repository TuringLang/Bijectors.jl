using NNlib

abstract type Constrained end

"""
    value(c::Constrained)

Return the value which is wrapped.
"""
value(c::Constrained)

struct Unconstrained{T, F} <: Constrained
    val::T
end
Unconstrained(val::T, f) where {T} = Unconstrained{T, f}(val)
Unconstrained(val::T) where {T} = Unconstrained{T, identity}(val)

value(c::Unconstrained{T, F}) where {T, F} = F(c.val)
f(::Unconstrained{T, F}) where {T, F} = F


struct RationalQuadraticSpline{T, D} <: Bijector{D}
    widths::T      # K widths
    heights::T     # K heights
    derivatives::T # (K - 1) derivatives

    function RationalQuadraticSpline(widths::T, heights::T, derivatives::T) where {T<:AbstractVector}
        # FIXME: add a `NoArgCheck` type and argument so we can circumvent if we want
        
        @assert length(widths) == length(heights) == length(derivatives)
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

function RationalQuadraticSpline(
    widths::Unconstrained{A, identity},
    heights::Unconstrained{A, identity},
    derivatives::Unconstrained{A, identity},
    B::T2
) where {T1, T2, A <: AbstractVector{T1}}
    # Using `NNLlinb.softax` instead of `StatsFuns.softmax` (which does inplace operations)
    return RationalQuadraticSpline(
        (cumsum(vcat([zero(T1)], NNlib.softmax(value(widths)))) .- 0.5) * 2 * B,
        (cumsum(vcat([zero(T1)], NNlib.softmax(value(heights)))) .- 0.5) * 2 * B,
        vcat([one(T1)], softplus.(value(derivatives)), [one(T1)])
    )
end

# TODO: implement transformations for the case where `Dim == 1`
function RationalQuadraticSpline(
    widths::Unconstrained{A, identity},
    heights::Unconstrained{A, identity},
    derivatives::Unconstrained{A, identity},
    B::T2
) where {T1, T2, A <: AbstractMatrix{T1}}
    # TODO: use `NNlib.softmax(widths; dims = 2)` when new version of `NNlib` is released
    
    # Using `NNLlinb.softax` instead of `StatsFuns.softmax` (which does inplace operations)
    return RationalQuadraticSpline(
        hcat([(cumsum(vcat([zero(T1)], NNlib.softmax(value(widths)[:, i]))) .- 0.5) * 2 * B for i = 1:size(value(widths), 2)]...),
        hcat([(cumsum(vcat([zero(T1)], NNlib.softmax(value(heights)[:, i]))) .- 0.5) * 2 * B for i = 1:size(value(heights), 2)]...),
        hcat([vcat([one(T1)], softplus.(value(derivatives)[:, i]), [one(T1)]) for i = 1:size(value(derivatives), 2)]...)
    )
end

function rqs_univariate(widths, heights, derivatives, x::Real)
    K = length(widths) - 1
    # @assert K == length(x) "length(x) should be the same length as the widhts/heights"

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
(b::RationalQuadraticSpline{<:AbstractVector, 0})(x::Real) = rqs_univariate(b.widths, b.heights, b.derivatives, x)
(b::RationalQuadraticSpline{<:AbstractVector, 0})(x::AbstractVector) = b.(x)

# multivariate
function (b::RationalQuadraticSpline{<:AbstractMatrix, 1})(x::AbstractVector)
    @assert length(x) == size(b.widths, 2) == size(b.heights, 2) == size(b.derivatives, 2)
    
    return [rqs_univariate(b.widths[:, i], b.heights[:, i], b.derivatives[:, i], x[i]) for i = 1:length(x)]
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

    # @assert isapprox(a1 * ξ^2 + a2 * ξ + a3, 0.0, atol=1e-8)
    
    return ξ * w + widths[k]
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

    numerator = s^2 * (derivatives[k + 1] * ξ^2 + 2 * s * ξ * (1 - ξ) + derivatives[k] * (1 - ξ)^2)
    denominator = s + (derivatives[k + 1] + derivatives[k] - 2 * s) * ξ * (1 - ξ)

    return log(numerator) - 2 * log(denominator)
end
logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::Real) = rqs_logabsdetjac(b.widths, b.heights, b.derivatives, x)
logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractVector) = logabsdetjac.(b, x)
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractVector)
    return sum([rqs_logabsdetjac(b.widths[:, i], b.heights[:, i], b.derivatives[:, i], x[i]) for i = 1:length(x)])
end
function logabsdetjac(b::RationalQuadraticSpline{<:AbstractMatrix, 1}, x::AbstractMatrix)
    return [logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)]
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
    numerator_jl = s^2 * (b.derivatives[k + 1] * ξ^2 + 2 * s * ξ * (1 - ξ) + b.derivatives[k] * (1 - ξ)^2)
    logjac = log(numerator_jl) - 2 * log(denominator)

    # y
    numerator_y = Δy * (s * ξ^2 + b.derivatives[k] * ξ * (1 - ξ))
    y = b.heights[k] + numerator_y / denominator

    return (rv = y, logabsdetjac = logjac)
end

# TODO: implement below
# function forward(b::RationalQuadraticSpline{<:AbstractVector, 0}, x::AbstractVector)
#     K = length(b.widths) - 1
    
#     # Find which bin `x` is in
#     ks = searchsortedfirst.(Ref(b.widths), x) .- 1
#     k_indicator = min.((ks .> K) + (ks .== 0), 1)
#     @info k_indicator
#     # if ks .> K || ks == 0
#     #     return (rv = x, logabsdetjac = zeros(eltype(x), length(x)))
#     # end

#     # Width
#     w = b.widths[ks .+ 1] - b.widths[ks]

#     # Slope
#     Δy = b.heights[ks .+ 1] - b.heights[ks]

#     @info w

#     # Recurring quantities
#     s = Δy ./ w
#     ξ = @. (x - b.widths[ks]) / w

#     # Re-used for both `logjac` and `y`
#     denominator = @. s + (b.derivatives[ks .+ 1] + b.derivatives[ks] - 2 * s) * ξ * (1 - ξ)

#     # logjac
#     numerator_jl = @. s^2 * (b.derivatives[ks .+ 1] * ξ^2 + 2 * s * ξ * (1 - ξ) + b.derivatives[ks] * (1 - ξ)^2)
#     logjac = @. log(numerator_jl) - 2 * log(denominator)

#     # y
#     numerator_y = @. Δy * (s * ξ^2 + b.derivatives[ks] * ξ * (1 - ξ))
#     y = @. b.heights[ks] + numerator_y / denominator

#     return (rv = y, logabsdetjac = logjac)
# end
