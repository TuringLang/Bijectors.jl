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
# PlanarLayer #
###############

# TODO: add docstring

# FIXME: using `TrackedArray` for the parameters, we end up with
# nested tracked structures; don't want this.
mutable struct PlanarLayer{T1,T2} <: Bijector{1}
    w::T1
    u::T1
    b::T2
end

function get_u_hat(u, w)
    # To preserve invertibility
    # From A.1
    return u .+ (planar_flow_m.(w' * u) .- w' * u) .* w ./ sum(abs2, w)
end

function PlanarLayer(dims::Int, container=Array)
    w = container(randn(dims, 1))
    u = container(randn(dims, 1))
    b = container(randn(1))
    return PlanarLayer(w, u, b)
end

planar_flow_m(x) = -1 + softplus(x)   # for planar flow from A.1
dtanh(x) = 1 - (tanh(x)) ^ 2         # for planar flow
ψ(z, w, b) = dtanh.(w' * z .+ b) .* w    # for planar flow from eq(11)

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

forward(flow::PlanarLayer, z::AbstractMatrix) = _forward(flow, z)

function forward(flow::PlanarLayer, z::AbstractVector{<: Real})
    res = _forward(flow, z)
    return (rv=vec(res.rv), logabsdetjac=res.logabsdetjac[1])
end

function (ib::Inversed{<:PlanarLayer})(y::AbstractMatrix{<: Real})
    flow = ib.orig
    u_hat = Bijectors.get_u_hat(flow.u, flow.w)

    # Define the objective functional; implemented with reference from A.1
    f(y) = alpha -> first((flow.w' * y) .- alpha .- (flow.w' * u_hat) .* tanh.(alpha .+ flow.b))
    
    # Run solver 
    alphas = [find_zero(f(y[:, i]), 0.0, Order16()) for i in 1:size(y, 2)]'
    z_para = (flow.w ./ norm(flow.w,2)) * alphas
    z_per = y - z_para - u_hat * tanh.(flow.w' * z_para .+ flow.b)

    return z_para + z_per
end

function (ib::Inversed{<:PlanarLayer})(y::AbstractVector{<:Real})
    return vec(ib(reshape(y, (length(y), 1))))
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inversed{<:PlanarLayer}) = false
