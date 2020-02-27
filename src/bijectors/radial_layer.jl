using LinearAlgebra
using Random
using NNlib: softplus
using Roots # for inverse

################################################################################
#                            Planar and Radial Flows                           #
#             Ref: Variational Inference with Normalizing Flows,               #
#               D. Rezende, S. Mohamed(2015) arXiv:1505.05770                  #
################################################################################

###############
# RadialLayer #
###############

mutable struct RadialLayer{T1 <: Real, T2 <: AbstractVector{<:Real}} <: Bijector{1}
    α_::T1
    β::T1
    z_0::T2
end

function RadialLayer(dims::Int, container=Array)
    α_ = randn()
    β = randn()
    z_0 = container(randn(dims))
    return RadialLayer(α_, β, z_0)
end

h(α, r) = 1 ./ (α .+ r)     # for radial flow from eq(14)
#dh(α, r) = .- (1 ./ (α .+ r)) .^ 2   # for radial flow; derivative of h()

# An internal version of transform that returns intermediate variables
function _transform(flow::RadialLayer, z::AbstractVecOrMat)
    return _radial_transform(flow.α_, flow.β, flow.z_0, z)
end
function _radial_transform(α_, β, z_0, z)
    α = softplus(α_)            # from A.2
    β_hat = -α + softplus(β)    # from A.2
    if z isa AbstractVector
        r = norm(z .- z_0)
    else
        r = vec(sqrt.(sum(abs2, z .- z_0; dims = 1)))
    end
    transformed = z .+ β_hat ./ (α .+ r') .* (z .- z_0)   # from eq(14)
    return (transformed = transformed, α = α, β_hat = β_hat, r = r)
end

(b::RadialLayer)(z::AbstractMatrix{<:Real}) = _transform(b, z).transformed
(b::RadialLayer)(z::AbstractVector{<:Real}) = vec(_transform(b, z).transformed)

function forward(flow::RadialLayer, z::AbstractVecOrMat)
    transformed, α, β_hat, r = _transform(flow, z)
    # Compute log_det_jacobian
    d = size(flow.z_0, 1)
    h_ = h(α, r)
    if transformed isa AbstractVector
        T = eltype(transformed)
    else
        T = typeof(vec(transformed))
    end
    log_det_jacobian::T = @. (
        (d - 1) * log(1 + β_hat * h_)
        + log(1 +  β_hat * h_ + β_hat * (- h_ ^ 2) * r)
    )   # from eq(14)
    return (rv = transformed, logabsdetjac = log_det_jacobian)
end

function (ib::Inverse{<:RadialLayer})(y::AbstractVector{<:Real})
    flow = ib.orig
    T = promote_type(eltype(flow.α_), eltype(flow.β), eltype(flow.z_0), eltype(y))
    TV = vectorof(T)
    α = softplus(flow.α_)            # from A.2
    β_hat = - α + softplus(flow.β)   # from A.2
    # Define the objective functional
    f(y) = r -> norm(y .- flow.z_0) - r * (1 + β_hat / (α + r))   # from eq(26)
    # Run solver
    rs::T = find_zero(f(y), zero(T), Order16())
    return (flow.z_0 .+ (y .- flow.z_0) ./ (1 .+ β_hat ./ (α .+ rs)))::TV
end
function (ib::Inverse{<:RadialLayer})(y::AbstractMatrix{<:Real})
    flow = ib.orig
    T = promote_type(eltype(flow.α_), eltype(flow.β), eltype(flow.z_0), eltype(y))
    TM = matrixof(T)
    α = softplus(flow.α_)            # from A.2
    β_hat = - α + softplus(flow.β)   # from A.2
    # Define the objective functional
    f(y) = r -> norm(y .- flow.z_0) - r * (1 + β_hat / (α + r))   # from eq(26)
    # Run solver
    init = @views vcat(find_zero(f(y[:,1]), zero(T), Order16()))
    rs::typeof(init) = mapreduce(vcat, drop(eachcol(y), 1); init = init) do c 
        find_zero(f(c), zero(T), Order16())
    end
    return (flow.z_0 .+ (y .- flow.z_0) ./ (1 .+ β_hat ./ (α .+ rs')))::TM
end

logabsdetjac(flow::RadialLayer, x::AbstractVecOrMat) = forward(flow, x).logabsdetjac
isclosedform(b::Inverse{<:RadialLayer}) = false
