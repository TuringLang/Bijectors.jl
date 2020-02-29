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

mutable struct PlanarLayer{T1<:AbstractVector{<:Real}, T2<:Real} <: Bijector{1}
    w::T1
    u::T1
    b::T2
end

function get_u_hat(u, w)
    # To preserve invertibility
    x = w' * u
    return u .+ (planar_flow_m(x) - x) .* w ./ sum(abs2, w)   # from A.1
end

function PlanarLayer(dims::Int, container=Array)
    w = container(randn(dims))
    u = container(randn(dims))
    b = randn()
    return PlanarLayer(w, u, b)
end

planar_flow_m(x) = -1 + softplus(x)   # for planar flow from A.1
ψ(z, w, b) = (1 .- tanh.(w' * z .+ b).^2) .* w    # for planar flow from eq(11)

# An internal version of transform that returns intermediate variables
function _transform(flow::PlanarLayer, z::AbstractVecOrMat)
    _planar_transform(flow.u, flow.w, flow.b, z)
end
function _planar_transform(u, w, b, z)
    u_hat = get_u_hat(u, w)
    transformed = z .+ u_hat .* tanh.(w' * z .+ b) # from eq(10)
    return (transformed = transformed, u_hat = u_hat)
end

(b::PlanarLayer)(z) = _transform(b, z).transformed

function forward(flow::PlanarLayer, z::AbstractVecOrMat)
    transformed, u_hat = _transform(flow, z)
    # Compute log_det_jacobian
    psi = ψ(z, flow.w, flow.b) .+ zero(eltype(u_hat))
    if psi isa AbstractVector
        T = eltype(psi)
    else
        T = typeof(vec(psi))
    end
    log_det_jacobian::T = log.(abs.(1.0 .+ psi' * u_hat)) # from eq(12)
    return (rv = transformed, logabsdetjac = log_det_jacobian)
end

function (ib::Inverse{<: PlanarLayer})(y::AbstractVector{<:Real})
    flow = ib.orig
    u_hat = get_u_hat(flow.u, flow.w)
    T = promote_type(eltype(flow.u), eltype(flow.w), eltype(flow.b), eltype(y))
    TV = vectorof(T)
    # Define the objective functional; implemented with reference from A.1
    f(y) = alpha -> (flow.w' * y) - alpha - (flow.w' * u_hat) * tanh(alpha + flow.b)
    # Run solver
    alpha::T = find_zero(f(y), zero(T), Order16())
    z_para::TV = (flow.w ./ norm(flow.w, 2)) .* alpha
    return (y .- u_hat .* tanh.(flow.w' * z_para .+ flow.b))::TV
end
function (ib::Inverse{<: PlanarLayer})(y::AbstractMatrix{<:Real})
    flow = ib.orig
    u_hat = get_u_hat(flow.u, flow.w)
    T = promote_type(eltype(flow.u), eltype(flow.w), eltype(flow.b), eltype(y))
    TM = matrixof(T)
    # Define the objective functional; implemented with reference from A.1
    f(y) = alpha -> (flow.w' * y) - alpha - (flow.w' * u_hat) * tanh(alpha + flow.b)
    # Run solver
    @views init = vcat(find_zero(f(y[:,1]), zero(T), Order16()))
    alpha::typeof(init) = mapreduce(vcat, drop(eachcol(y), 1); init = init) do c 
        find_zero(f(c), zero(T), Order16())
    end
    z_para::TM = (flow.w ./ norm(flow.w, 2)) .* alpha'
    return (y .- u_hat .* tanh.(flow.w' * z_para .+ flow.b))::TM
end

function matrixof(::Type{Vector{T}}) where {T <: Real}
    return Matrix{T}
end
function matrixof(::Type{T}) where {T <: Real}
    return Matrix{T}
end
function vectorof(::Type{Matrix{T}}) where {T <: Real}
    return Vector{T}
end
function vectorof(::Type{T}) where {T <: Real}
    return Vector{T}
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inverse{<:PlanarLayer}) = false
