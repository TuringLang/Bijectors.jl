using Distributions
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
# PlanarLayer #
###############

mutable struct PlanarLayer{T1,T2} <: Bijector
    w::T1
    u::T1
    b::T2
end

function get_u_hat(u, w)
    # To preserve invertibility
    return (
        u + (planar_flow_m(w' * u) - w' * u)[1]
        * w / (norm(w[:,1],2) ^ 2)
    )   # from A.1
end

function PlanarLayer(dims::Int, container=Array)
    w = container(randn(dims, 1))
    u = container(randn(dims, 1))
    b = container(randn(1))
    return PlanarLayer(w, u, b)
end

planar_flow_m(x) = -1 .+ softplus.(x)   # for planar flow from A.1
dtanh(x) = 1 .- (tanh.(x)) .^ 2         # for planar flow
ψ(z, w, b) = dtanh(w' * z .+ b) .* w    # for planar flow from eq(11)

# An internal version of transform that returns intermediate variables
function _transform(flow::PlanarLayer, z)
    u_hat = get_u_hat(flow.u, flow.w)
    transformed = z + u_hat * tanh.(flow.w' * z .+ flow.b) # from eq(10)
    return (transformed=transformed, u_hat=u_hat)
end

(b::PlanarLayer)(z) = _transform(b, z).transformed

function _forward(flow::PlanarLayer, z)
    transformed, u_hat = _transform(flow, z)
    # Compute log_det_jacobian
    psi = ψ(z, flow.w, flow.b)
    log_det_jacobian = log.(abs.(1.0 .+ psi' * u_hat))          # from eq(12)
    return (rv=transformed, logabsdetjac=vec(log_det_jacobian)) # from eq(10)
end

forward(flow::PlanarLayer, z) = _forward(flow, z)

function forward(flow::PlanarLayer, z::AbstractVector{<: Real})
    res = _forward(flow, z)
    return (rv=res.rv, logabsdetjac=res.logabsdetjac[1])
end


function (ib::Inversed{<: PlanarLayer})(y::AbstractMatrix{<: Real})
    flow = ib.orig
    u_hat = get_u_hat(flow.u, flow.w)
    # Define the objective functional; implemented with reference from A.1
    f(y) = alpha -> (flow.w' * y)[1] - alpha - (flow.w' * u_hat)[1] * tanh(alpha+flow.b[1])
    # Run solver 
    alphas_ = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y, 2)]
    alphas = alphas_'
    z_para = (flow.w ./ norm(flow.w,2)) * alphas
    z_per = y - z_para - u_hat * tanh.(flow.w' * z_para .+ flow.b)

    return z_para + z_per
end

function (ib::Inversed{<: PlanarLayer})(y::AbstractVector{<: Real})
    return vec(ib(reshape(y, (length(y), 1))))
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac

###############
# RadialLayer #
###############

# FIXME: using `TrackedArray` for the parameters, we end up with
# nested tracked structures; don't want this.
mutable struct RadialLayer{T1,T2} <: Bijector
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
    α = softplus(flow.α_[1])            # from A.2
    β_hat = -α + softplus(flow.β[1])    # from A.2
    r = norm.([z[:,i] .- flow.z_0 for i in 1:size(z, 2)], 2)'
    transformed = z + β_hat .* h(α, r) .* (z .- flow.z_0)   # from eq(14)
    return (transformed=transformed, α=α, β_hat=β_hat, r=r)
end

(b::RadialLayer)(z) = _transform(b, z).transformed

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

function forward(flow::RadialLayer, z::AbstractVector{<: Real})
    res = forward(flow, z)
    return (rv=res.rv, logabsdetjac=res.logabsdetjac[1])
end

# function inv(flow::RadialLayer, y)
function (ib::Inversed{<: RadialLayer})(y)
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

function (ib::Inversed{<: RadialLayer})(y::AbstractVector{<: Real})
    return vec(ib(reshape(y, (length(y), 1))))
end

logabsdetjac(flow::RadialLayer, x) = forward(flow, x).logabsdetjac
