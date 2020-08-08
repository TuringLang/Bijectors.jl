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

mutable struct PlanarLayer{T1<:AbstractVector{<:Real}, T2<:Union{Real, AbstractVector{<:Real}}} <: Bijector{1}
    w::T1
    u::T1
    b::T2
end
function Base.:(==)(b1::PlanarLayer, b2::PlanarLayer)
    return b1.w == b2.w && b1.u == b2.u && b1.b == b2.b
end

function get_u_hat(u, w)
    # To preserve invertibility
    x = w' * u
    return u .+ (planar_flow_m(x) - x) .* w ./ sum(abs2, w)   # from A.1
end

function PlanarLayer(dims::Int, wrapper=identity)
    w = wrapper(randn(dims))
    u = wrapper(randn(dims))
    b = wrapper(randn(1))
    return PlanarLayer(w, u, b)
end

planar_flow_m(x) = -1 + softplus(x)   # for planar flow from A.1
ψ(z, w, b) = (1 .- tanh.(w' * z .+ b).^2) .* w    # for planar flow from eq(11)

# An internal version of transform that returns intermediate variables
function _transform(flow::PlanarLayer, z::AbstractVecOrMat)
    return _planar_transform(flow.u, flow.w, first(flow.b), z)
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
    psi = ψ(z, flow.w, first(flow.b)) .+ zero(eltype(u_hat))
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
    w = flow.w
    b = first(flow.b)
    u_hat = get_u_hat(flow.u, w)

    # Find the scalar ``alpha`` from A.1.
    wt_y = dot(w, y)
    wt_u_hat = dot(w, u_hat)
    alpha = find_alpha(y, wt_y, wt_u_hat, b)

    return y .- u_hat .* tanh(alpha * norm(w, 2) + b)
end

function (ib::Inverse{<: PlanarLayer})(y::AbstractMatrix{<:Real})
    flow = ib.orig
    w = flow.w
    b = first(flow.b)
    u_hat = get_u_hat(flow.u, flow.w)

    # Find the scalar ``alpha`` from A.1 for each column.
    wt_u_hat = dot(w, u_hat)
    alphas = mapvcat(eachcol(y)) do c
        find_alpha(c, dot(w, c), wt_u_hat, b)
    end

    return y .- u_hat .* tanh.(alphas' .* norm(w, 2) .+ b)
end

"""
    find_alpha(y::AbstractVector{<:Real}, wt_y, wt_u_hat, b)

Compute an (approximate) real-valued solution ``α`` to the equation
```math
wt_y = α + wt_u_hat tanh(α + b)
```

The uniqueness of the solution is guaranteed since ``wt_u_hat ≥ -1``.
For details see appendix A.1 of the reference.

# References

D. Rezende, S. Mohamed (2015): Variational Inference with Normalizing Flows.
arXiv:1505.05770
"""
function find_alpha(y::AbstractVector{<:Real}, wt_y, wt_u_hat, b)
    # Compute the initial bracket ((-Inf, 0) or (0, Inf))
    f0 = wt_u_hat * tanh(b) - wt_y
    zero_f0 = zero(f0)
    if f0 < zero_f0
        initial_bracket = (zero_f0, oftype(f0, Inf))
    else
        initial_bracket = (oftype(f0, -Inf), zero_f0)
    end
    alpha = find_zero(initial_bracket) do x
        x + wt_u_hat * tanh(x + b) - wt_y
    end

    return alpha
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inverse{<:PlanarLayer}) = false
