# Code adapted from Flux.jl:
# https://github.com/FluxML/Flux.jl/blob/68ba6e4e2fa4b86e2fef8dc6d0a5d795428a6fac/src/layers/normalise.jl#L117-L206
# License: https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md

"""
InvertibleBatchNorm(β, logγ, μ, σ², ϵ::AbstractFloat, momentum::AbstractFloat, active::Bool)

β, logγ - learned parameters

`InvertibleBatchNorm` computes the mean and variance for each each `NxD` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters). 

See [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf).

"""
mutable struct InvertibleBatchNorm{T1,T2,T3} <: Bijector
    β::T1
    logγ::T1
    μ::T2  # moving mean
    σ²::T2 # moving st
    ϵ::T3
    momentum::T3
    active::Bool # true when training
end

InvertibleBatchNorm(dims::Int,
                    container=Array; 
                    ϵ::AbstractFloat=1f-5,
                    momentum::AbstractFloat=0.1f0) = (
                    InvertibleBatchNorm(
                            container(zeros(dims, 1)),
                            container(zeros(dims, 1)),
                            zeros(dims, 1),
                            ones(dims, 1),
                            ϵ,
                            momentum,
                            true
                    ))

function affinesize(x)
    dims = length(size(x))
    channels = size(x, dims)
    affinesize = ones(Int, dims)
    affinesize[end-1] = channels
    return affinesize
end

logabsdetjac(
    t::T,
    x
) where{T<:InvertibleBatchNorm} = forward(t, x).logabsdetjac

function _compute_μ_σ²(t::InvertibleBatchNorm, x)
    @assert size(x, ndims(x)) == 
    length(t.μ) "`InvertibleBatchNorm` expected $(length(t.μ)) channels, got $(
        size(x,ndims(x)))"
    as = affinesize(x)
    m = size(x)[1]
    γ = exp.(reshape(t.logγ, as...))
    β = reshape(t.β, as...)
    if !t.active
        μ = reshape(t.μ, as...)
        σ² = reshape(t.σ², as...)
        ϵ = t.ϵ
    else
        Tx = eltype(x)
        axes = 1
        μ = mean(x, dims=axes)'
        σ² = sum((x .- μ') .^ 2, dims=axes)' ./ m
    end
    return μ, σ²
end

function forward(t::InvertibleBatchNorm, x)
    μ, σ² = _compute_μ_σ²(t, x)
    as = affinesize(x)
    m = size(x)[1]
    γ = exp.(reshape(t.logγ, as...))
    β = reshape(t.β, as...)
    Tx = eltype(x)
    if t.active
        ϵ = convert(Tracker.data(Tx), t.ϵ)
        # Update moving mean/std
        mtm = convert(Tracker.data(Tx), t.momentum)
        t.μ = (1 - mtm) .* t.μ .+ mtm .* reshape(Tracker.data(μ), :)
        t.σ² = ((1 - mtm) .* t.σ² .+ (mtm * m / (m - 1)) 
                .* reshape(Tracker.data(σ²), :))
    end
    x̂ = (x .- μ') ./ sqrt.(σ² .+ t.ϵ)'
    
    logabsdetjac = ((sum(t.logγ - log.(σ² .+ t.ϵ) / 2))
                    .* typeof(Tracker.data(x))(ones(Float32, size(x, 1))'))'

    return (rv=γ' .* x̂ .+ β', logabsdetjac=logabsdetjac)
end

(b::InvertibleBatchNorm)(z) = forward(b, z).rv

function forward(it::Inversed{T}, y) where {T<:InvertibleBatchNorm}
    t = inv(it)
    @assert t.active ==
     false "`forward(::Inversed{InvertibleBatchNorm})` is only available in test mode 
     but not in training mode."
    as = affinesize(y)
    γ = exp.(reshape(t.logγ, as...))
    β = reshape(t.β, as...)
    μ = reshape(t.μ, as...)
    σ² = reshape(t.σ², as...)

    ŷ = (y .- β') ./ γ'
    x = ŷ .* sqrt.(σ²' .+ t.ϵ) .+ μ'

    return (rv=x, logabsdetjac=-logabsdetjac(t, x))
end

(b::Inversed{<: InvertibleBatchNorm})(z) = forward(b, z).rv
