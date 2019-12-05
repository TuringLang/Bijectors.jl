using LinearAlgebra
using Random
using StatsFuns: softplus
using Roots # for inverse

################################################################################
#                            Planar and Radial Flows                           #
#             Ref: Variational Inference with Normalizing Flows,               #
#               D. Rezende, S. Mohamed(2015) arXiv:1505.05770                  #
################################################################################

###############
# RadialLayer #
###############

# FIXME: using `TrackedArray` for the parameters, we end up with
# nested tracked structures; don't want this.
mutable struct RadialLayer{T1,T2} <: Bijector{1}
    α_::T1
    β::T1
    z_0::T2
end

function RadialLayer(dims::Int, container=Array)
    α_ = container(randn(1))
    β = container(randn(1))
    z_0 = container(randn(dims, 1))
    return RadialLayer(α_, β, z_0)
end

h(α, r) = 1 ./ (α .+ r)     # for radial flow from eq(14)
dh(α, r) = - h(α, r) .^ 2   # for radial flow; derivative of h()

# An internal version of transform that returns intermediate variables
function _transform(flow::RadialLayer, z)
    # from A.2
    α = if flow.α_ isa CuArray
        softplus_gpu.(flow.α_)
    else
        softplus(flow.α_[1])
    end
    # from A.2
    β_hat = if flow.β isa CuArray
        -α + softplus_gpu.(flow.β)
    else
        -α + softplus(flow.β[1])
    end
    r = sqrt.(sum((z .- flow.z_0).^2; dims = 1))
    transformed = z + β_hat .* h(α, r) .* (z .- flow.z_0)   # from eq(14)
    return (transformed=transformed, α=α, β_hat=β_hat, r=r)
end

(b::RadialLayer)(z::AbstractMatrix{<:Real}) = _transform(b, z).transformed
(b::RadialLayer)(z::AbstractVector{<:Real}) = vec(_transform(b, z).transformed)

function _forward(flow::RadialLayer, z)
    transformed, α, β_hat, r = _transform(flow, z)
    # Compute log_det_jacobian
    d = size(flow.z_0, 1)
    h_ = h(α, r)
    log_det_jacobian = @. (
        (d - 1) * log(1.0 + β_hat * h_)
        + log(1.0 +  β_hat * h_ + β_hat * (- h_ ^ 2) * r)
    )   # from eq(14)
    return (rv=transformed, logabsdetjac=vec(log_det_jacobian))
end

forward(flow::RadialLayer, z) = _forward(flow, z)

function forward(flow::RadialLayer, z::AbstractVector{<:Real})
    res = _forward(flow, z)
    return (rv=vec(res.rv), logabsdetjac=res.logabsdetjac[1])
end

function (ib::Inversed{<:RadialLayer})(y)
    flow = ib.orig
    α = softplus(flow.α_[1])            # from A.2
    β_hat = - α + softplus(flow.β[1])   # from A.2
    # Define the objective functional
    f(y) = r -> norm(y - flow.z_0, 2) - r * (1 + β_hat / (α + r))   # from eq(26)
    # Run solver 
    rs = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y, 2)]'    # from A.2
    z_hat = (y .- flow.z_0) ./ (rs .* (1 .+ β_hat ./ (α .+ rs)))            # from eq(25)
    z = flow.z_0 .+ rs .* z_hat # from A.2
    return z
end

function (ib::Inversed{<: RadialLayer})(y::AbstractVector{<:Real})
    return vec(ib(reshape(y, (length(y), 1))))
end

logabsdetjac(flow::RadialLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inversed{<:RadialLayer}) = false
