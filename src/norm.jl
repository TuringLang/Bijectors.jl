# Code adapted from Flux.jl:
# https://github.com/FluxML/Flux.jl/blob/68ba6e4e2fa4b86e2fef8dc6d0a5d795428a6fac/src/layers/normalise.jl#L117-L206
# License: https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md

mutable struct BatchNormFlow{T1,T2} <: Bijector
    β::T1
    logγ::T1
    μ  # moving mean
    σ² # moving st
    ϵ::T2
    momentum::T2
    active::Bool # true when training
end

BatchNormFlow(dims::Int, container=Array; ϵ=1f-5, momentum=0.1f0) = BatchNormFlow(
    container(zeros(Float32, dims)),
    container(zeros(Float32, dims)),
    zeros(Float32, dims),
    ones(Float32, dims),
    ϵ,
    momentum,
    true
)

function affinesize(x)
    dims = length(size(x))
    channels = size(x, dims-1)
    affinesize = ones(Int, dims)
    affinesize[end-1] = channels
    return affinesize
end

logabsdetjacob(
    t::T,
    x;
    σ²=reshape(t.σ², affinesize(x)...)
) where {T<:BatchNormFlow} =  (sum(t.logγ - log.(σ² .+ t.ϵ) / 2)) .* typeof(Flux.data(x))(ones(Float32, size(x, 2))')

function _transform(t::BatchNormFlow, x)
     @assert size(x, ndims(x) - 1) == length(t.μ) "`BatchNormFlow` expected $(length(t.μ)) channels, got $(size(x, ndims(x) - 1))"
    as = affinesize(x)
    m = prod(size(x)[1:end-2]) * size(x)[end]
    γ = exp.(reshape(t.logγ, as...))
    β = reshape(t.β, as...)
    if !t.active
        μ = reshape(t.μ, as...)
        σ² = reshape(t.σ², as...)
        ϵ = t.ϵ
    else
        Tx = eltype(x)
        dims = length(size(x))
        axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
        μ = mean(x, dims=axes)
        σ² = sum((x .- μ) .^ 2, dims=axes) ./ m
        ϵ = Flux.data(convert(Tx, t.ϵ))
        # Update moving mean/std
        mtm = Flux.data(convert(Tx, t.momentum))
        t.μ = (1 - mtm) .* t.μ .+ mtm .* reshape(Flux.data(μ), :)
        t.σ² = (1 - mtm) .* t.σ² .+ (mtm * m / (m - 1)) .* reshape(Flux.data(σ²), :)
    end

    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    return (rv=γ .* x̂ .+ β, σ²=σ²)
end

(b::BatchNormFlow)(z) = _transform(b, z).rv

function _forward(t::BatchNormFlow, x)
    rv, σ² = _transform(t, x)
    return (rv=rv, logabsdetjacob=logabsdetjacob(t, x; σ²=σ²))
end

forward(flow::BatchNormFlow, z) = _forward(flow, z)

# TODO: make this function take kw argument `σ²`
logabsdetjacob(it::Inversed{T}, y) where {T<:BatchNormFlow} = (xsimilar = y; -logabsdetjacob(inv(it), xsimilar))

function forward(it::Inversed{T}, y) where {T<:BatchNormFlow}
    t = inv(it)
    @assert t.active == false "`forward(::Inversed{BatchNormFlow})` is only available in test mode but not in training mode."
    as = affinesize(y)
    γ = exp.(reshape(t.logγ, as...))
    β = reshape(t.β, as...)
    μ = reshape(t.μ, as...)
    σ² = reshape(t.σ², as...)

    ŷ = (y .- β) ./ γ
    x = ŷ .* sqrt.(σ² .+ t.ϵ) .+ μ
    return (rv=x, logabsdetjacob=logabsdetjacob(it, x))
end

(b::Inversed{<: BatchNormFlow})(z) = forward(b, z).rv
