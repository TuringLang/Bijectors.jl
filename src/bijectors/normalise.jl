using Statistics: mean

# Code adapted from Flux.jl
# Ref: https://github.com/FluxML/Flux.jl/blob/master/src/layers/normalise.jl#L93-L177
# License: https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md

istraining() = false

mutable struct InvertibleBatchNorm{T1,T2,T3} <: Bijector
    b       ::  T1  # bias
    logs    ::  T1  # log-scale
    m       ::  T2  # moving mean
    v       ::  T2  # moving variance
    eps     ::  T3
    mtm     ::  T3  # momentum
end
function Base.:(==)(b1::InvertibleBatchNorm, b2::InvertibleBatchNorm)
    return  b1.b == b2.b && 
            b1.logs == b2.logs &&
            b1.m == b2.m &&
            b1.v == b2.v &&
            b1.eps == b2.eps &&
            b1.mtm == b2.mtm
end

function InvertibleBatchNorm(
    chs::Int;
    eps::T=1f-5,
    mtm::T=1f-1,
) where {T<:AbstractFloat}
    return InvertibleBatchNorm(
        zeros(T, chs),
        zeros(T, chs),  # logs = 0 means s = 1
        zeros(T, chs),
         ones(T, chs),
        eps,
        mtm,
    )
end

# define numerical parameters
# TODO: replace with `Functors.@functor InvertibleBatchNorm (b, logs)` when
# https://github.com/FluxML/Functors.jl/pull/7 is merged
function Functors.functor(::Type{<:InvertibleBatchNorm}, x)
    function reconstruct_invertiblebatchnorm(xs)
        return InvertibleBatchNorm(xs.b, xs.logs, x.m, x.v, x.eps, x.mtm)
    end
    return (b = x.b, logs = x.logs), reconstruct_invertiblebatchnorm
end

function with_logabsdet_jacobian(bn::InvertibleBatchNorm, x)
    dims = ndims(x)
    size(x, dims - 1) == length(bn.b) ||
        error("InvertibleBatchNorm expected $(length(bn.b)) channels, got $(size(x, dims - 1))")
    channels = size(x, dims - 1)
    as = ntuple(i -> i == ndims(x) - 1 ? size(x, i) : 1, dims)
    logs = reshape(bn.logs, as...)
    s = exp.(logs)
    b = reshape(bn.b, as...)
    if istraining()
        n = div(prod(size(x)), channels)
        axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
        m = mean(x, dims = axes)
        v = sum((x .- m) .^ 2, dims = axes) ./ n
        # Update moving mean and variance
        mtm = bn.mtm
        T = eltype(bn.m)
        bn.m = (1 - mtm) .* bn.m .+ mtm .* T.(reshape(m, :))
        bn.v = (1 - mtm) .* bn.v .+ (mtm * n / (n - 1)) .* T.(reshape(v, :))
    else
        m = reshape(bn.m, as...)
        v = reshape(bn.v, as...)
    end

    result = s .* (x .- m) ./ sqrt.(v .+ bn.eps) .+ b
    logabsdetjac = (
        fill(sum(logs - log.(v .+ bn.eps) / 2), size(x, dims))
    )
    return (result=result, logabsdetjac=logabsdetjac)
end

logabsdetjac(bn::InvertibleBatchNorm, x) = forward(bn, x).logabsdetjac
transform(bn::InvertibleBatchNorm, x) = forward(bn, x).result

function with_logabsdet_jacobian(invbn::Inverse{<:InvertibleBatchNorm}, y)
    @assert !istraining() "`with_logabsdet_jacobian(::Inverse{InvertibleBatchNorm})` is only available in test mode."
    dims = ndims(y)
    as = ntuple(i -> i == ndims(y) - 1 ? size(y, i) : 1, dims)
    bn = inverse(invbn)
    s = reshape(exp.(bn.logs), as...)
    b = reshape(bn.b, as...)
    m = reshape(bn.m, as...)
    v = reshape(bn.v, as...)

    x = (y .- b) ./ s .* sqrt.(v .+ bn.eps) .+ m
    return (result=x, logabsdetjac=-logabsdetjac(bn, x))
end

transform(bn::Inverse{<:InvertibleBatchNorm}, y) = forward(bn, y).result

function Base.show(io::IO, l::InvertibleBatchNorm)
    print(io, "InvertibleBatchNorm($(join(size(l.b), ", ")))")
end
