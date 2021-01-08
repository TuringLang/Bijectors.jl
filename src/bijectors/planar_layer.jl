using LinearAlgebra
using Random
using NNlib: softplus

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
    log_det_jacobian::T = log.(abs.(1 .+ psi' * u_hat)) # from eq(12)
    return (rv = transformed, logabsdetjac = log_det_jacobian)
end

function (ib::Inverse{<:PlanarLayer})(y::AbstractVector{<:Real})
    flow = ib.orig
    w = flow.w
    b = first(flow.b)
    u_hat = get_u_hat(flow.u, w)

    # Find the scalar ``alpha`` from A.1.
    wt_y = dot(w, y)
    wt_u_hat = dot(w, u_hat)
    alpha = find_alpha(wt_y, wt_u_hat, b)

    return y .- u_hat .* tanh(alpha + b)
end

function (ib::Inverse{<:PlanarLayer})(y::AbstractMatrix{<:Real})
    flow = ib.orig
    w = flow.w
    b = first(flow.b)
    u_hat = get_u_hat(flow.u, flow.w)

    # Find the scalar ``alpha`` from A.1 for each column.
    wt_u_hat = dot(w, u_hat)
    alphas = mapvcat(eachcol(y)) do c
        find_alpha(dot(w, c), wt_u_hat, b)
    end

    return y .- u_hat .* tanh.(reshape(alphas, 1, :) .+ b)
end

"""
    find_alpha(wt_y, wt_u_hat, b)

Compute an (approximate) real-valued solution ``α̂`` to the equation
```math
wt_y = α + wt_u_hat tanh(α + b)
```

The uniqueness of the solution is guaranteed since ``wt_u_hat ≥ -1``.
For details see appendix A.1 of the reference.

# Initial bracket

For all ``α``, we have
```math
α - |wt_u_hat| - wt_y \\leq α + wt_u_hat tanh(α + b) - wt_y \\leq α + |wt_u_hat| - wt_y.
```
Thus
```math
α̂ - |wt_u_hat| - wt_y \\leq 0 \\leq α̂ + |wt_u_hat| - wt_y,
```
which implies ``α̂ ∈ [wt_y - |wt_u_hat|, wt_y + |wt_u_hat|]``.

# References

D. Rezende, S. Mohamed (2015): Variational Inference with Normalizing Flows.
arXiv:1505.05770
"""
function find_alpha(wt_y, wt_u_hat, b)
    # Compute the initial bracket.
    _wt_y, _wt_u_hat, _b = promote(wt_y, wt_u_hat, b)
    initial_bracket = (_wt_y - abs(_wt_u_hat), _wt_y + abs(_wt_u_hat))

    prob = NonlinearSolve.NonlinearProblem{false}(initial_bracket) do α, _
        α + _wt_u_hat * tanh(α + _b) - _wt_y
    end
    alpha = NonlinearSolve.solve(prob, NonlinearSolve.Falsi()).left
    return alpha
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inverse{<:PlanarLayer}) = false
