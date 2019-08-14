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

mutable struct PlanarLayer{T1,T2} <: Bijector
    w::T1
    u::T1
    u_hat::T1
    b::T2
end

function get_u_hat(u, w)
    # To preserve invertibility
    return (
        u + (planar_flow_m(transpose(w) * u) - transpose(w) * u)[1]
        * w / (norm(w[:,1],2) ^ 2)
    ) # from A.1
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

planar_flow_m(x) = -1 .+ softplus.(x) # for planar flow from A.1
dtanh(x) = 1 .- (tanh.(x)) .^ 2 # for planar flow
ψ(z, w, b) = dtanh(transpose(w) * z .+ b) .* w # for planar flow from eq(11)

function transform(flow::PlanarLayer, z)
    return z + flow.u_hat * tanh.(transpose(flow.w) * z .+ flow.b) # from eq(10)
end

function forward(flow::T, z) where {T<:PlanarLayer}
    update_u_hat!(flow)
    # Compute log_det_jacobian
    psi = ψ(z, flow.w, flow.b)
    log_det_jacobian = log.(abs.(1.0 .+ transpose(psi) * flow.u_hat)) # from eq(12)
    transformed = z + flow.u_hat * tanh.(transpose(flow.w) * z .+ flow.b)
    return (rv=transformed, logabsdetjacob=log_det_jacobian) # from eq(10)
end

function inv(flow::PlanarLayer, y)
    # Implemented with reference from A.1
    function f(y)
        return loss(alpha) = (
                (transpose(flow.w) * y)[1] - alpha
                - (transpose(flow.w) * flow.u_hat)[1]
                * tanh(alpha+flow.b[1])
            )
    end
    alphas_ = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y, 2)]
    alphas = transpose(alphas_)
    z_para = (flow.w ./ norm(flow.w,2)) * alphas
    z_per = (
            y - z_para - flow.u_hat * tanh.(
                                    transpose(flow.w) * z_para
                                    .+ flow.b
            )
    )

    return z_para + z_per
end

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



h(α, r) = 1 ./ (α .+ r) # for radial flow from eq(14)
dh(α, r) = - h(α, r) .^ 2 # for radial flow, derivative of h()


function transform(flow::RadialLayer, z)
    α = softplus(flow.α_[1]) # from A.2
    β_hat = -α + softplus(flow.β[1]) # from A.2
    r = transpose(norm.([z[:,i] .- flow.z_0 for i in 1:size(z, 2)], 1))
    return z + β_hat .* h(α, r) .* (z .- flow.z_0) # from eq(14)
end



function forward(flow::T, z) where {T<:RadialLayer}
    # Compute log_det_jacobian
    α = softplus(flow.α_[1]) # from A.2
    β_hat = -α + softplus(flow.β[1]) # from A.2
    r = transpose(norm.([z[:,i] .- flow.z_0 for i in 1:size(z, 2)], 1))
    transformed = z + β_hat .* h(α, r) .* (z .- flow.z_0) # from eq(14)

    d = size(flow.z_0, 1)
    h_ = h(α, r)
    log_det_jacobian = @. (
        (d-1) * log(1.0 + β_hat * h_)
        + log(1.0 +  β_hat * h_ + β_hat * (- h_ ^ 2) * r)
    ) # from eq(14)
    return (rv=transformed, logabsdetjacob=log_det_jacobian)
end

function inv(flow::RadialLayer, y)
    α = softplus(flow.α_[1]) # from A.2
    β_hat = - α + softplus(flow.β[1]) # from A.2
    function f(y)
        # From eq(26)
        return loss(r) = (
                        norm(y - flow.z_0, 2)
                        - r * (1 + β_hat / (α + r))
                        )
    end
    rs_ = [find_zero(f(y[:,i:i]), 0.0, Order16()) for i in 1:size(y, 2)] # A.2
    rs = transpose(rs_)
    z_hat = (y .- flow.z_0) .* (rs .* (1 .+ β_hat ./ (α .+ rs)) ) # from eq(25)
    z = flow.z_0 .+ rs .* z_hat # from A.2
    return z
end
