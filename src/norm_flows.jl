using Distributions
using LinearAlgebra
using Random
using StatsFuns: softplus
using Roots, LinearAlgebra # for inverse

################################################################################
#                            Planar and Radial Flows                           #
#             Ref: Variational Inference with Normalizing Flows,               #
#               D. Rezende, S. Mohamed(2015) arXiv:1505.05770                  #
################################################################################

mutable struct PlanarLayer <: Bijector
    w
    u
    u_hat
    b
end

function get_u_hat(u, w)
    # to preserve invertibility
    u_hat = (
        u + (planar_flow_m(transpose(w) * u) - transpose(w) * u)[1]
        * w/(norm(w[:,1],2) ^ 2)
        )
end

function update_u_hat!(flow::PlanarLayer)
    flow.u_hat = get_u_hat(flow.u, flow.w)
end


function PlanarLayer(dims::Int, container=Array)
    w = container(randn(dims, 1))
    u = container(randn(dims, 1))
    b = container(randn(1))
    u_hat = get_u_hat(u, w)
    return PlanarLayer(w, u, u_hat, b)
end

planar_flow_m(x) = -1 .+ softplus.(x) # for planar flow
dtanh(x) = 1 .- (tanh.(x)).^2 # for planar flow

function transform(flow::PlanarLayer, z)
    return z + flow.u_hat*tanh.(transpose(flow.w)*z .+ flow.b)
end

function forward(flow::T, z) where {T<:PlanarLayer}
    update_u_hat!(flow)
    # Compute log_det_jacobian
    psi = ψ(z, flow.w, flow.b)
    log_det_jacobian = log.(abs.(1.0 .+ transpose(psi)*flow.u_hat))

    return (rv=transformed, logabsdetjacob=log_det_jacobian)
end

function inv(flow::PlanarLayer, y)
    function f(y)
        return loss(alpha) = (
                (transpose(flow.w.data)*y)[1] - alpha
                - (transpose(flow.w.data)*flow.u_hat.data)[1]
                * tanh(alpha+flow.b.data[1])
            )
    end
    alphas_ = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y)[2]]
    alphas = transpose(alphas)
    z_para = (flow.w.data ./norm(flow.w.data,2)) * alphas
    z_per = (
            y - z_para - flow.u_hat.data*tanh.(
                                    transpose(flow.w.data) * z_para
                                    .+ flow.b.data
                                    )
            )

    return z_para+z_per
end

mutable struct RadialLayer <: Bijector
    α_
    β
    z_0
end

function RadialLayer(dims::Int, container=Array)
    α_ = container(randn(1))
    β = container(randn(1))
    z_0 = container(randn(dims, 1))
    return RadialLayer(α_, β, z_0)
end



ψ(z, w, b) = dtanh(transpose(w) * z .+ b) .* w # for planar flow
h(α, r) = 1 ./ (α .+ r) # for radial flow
dh(α, r) = -h(α, r).^2 # for radial flow


function transform(flow::RadialLayer, z)
    α = softplus(flow.α_[1])
    β_hat = -α + softplus(flow.β[1])
    r = norm.(z .- flow.z_0, 1)
    return z + β_hat .* h(α, r) .* (z .- flow.z_0)
end



function forward(flow::T, z) where {T<:RadialLayer}
    # compute log_det_jacobian
    transformed = transform(flow, z)
    α = softplus(flow.α_[1])
    β_hat = -α + softplus(flow.β[1])
    r = transpose(norm.([z[:,i] .- flow.z_0[:,:] for i in 1:size(z)[2]], 1))
    d = size(flow.z_0)[1]
    h_ = h(α, r)
    h_ = h(α, r)
    log_det_jacobian = @. (
        (d-1) * log(1.0 + β_hat * h_)
        + log(1.0 +  β_hat * h_ + β_hat * (- h_^2) * r)
    )
    return (rv=transformed, logabsdetjacob=log_det_jacobian)
end

function inv(flow::RadialLayer, y)
    α = softplus(flow.α_.data[1])
    β_hat = -α + softplus(flow.β.data[1])
    function f(y)
        return loss(r) = (
                        norm(y - flow.z_not.data, 2)
                        - r * (1 + β_hat / (α + r))
                        )
    end
    rs_ = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y)[2]]
    rs = transpose(alphas)
    z = (y.-flow.z_not.data) .* (rs .* (1 .+ β_hat ./ (α .+ rs)) )
    return z
end
