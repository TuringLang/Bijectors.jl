################################################################################
#                            Planar and Radial Flows                           #
#             Ref: Variational Inference with Normalizing Flows,               #
#               D. Rezende, S. Mohamed(2015) arXiv:1505.05770                  #
################################################################################

###############
# RadialLayer #
###############

mutable struct RadialLayer{T1<:Union{Real, AbstractVector{<:Real}}, T2<:AbstractVector{<:Real}} <: Bijector{1}
    α_::T1
    β::T1
    z_0::T2
end
function Base.:(==)(b1::RadialLayer, b2::RadialLayer)
    return b1.α_ == b2.α_ && b1.β == b2.β && b1.z_0 == b2.z_0
end

function RadialLayer(dims::Int, wrapper=identity)
    α_ = wrapper(randn(1))
    β = wrapper(randn(1))
    z_0 = wrapper(randn(dims))
    return RadialLayer(α_, β, z_0)
end

# all fields are numerical parameters
Functors.@functor RadialLayer

h(α, r) = 1 ./ (α .+ r)     # for radial flow from eq(14)
#dh(α, r) = .- (1 ./ (α .+ r)) .^ 2   # for radial flow; derivative of h()

# An internal version of transform that returns intermediate variables
function _transform(flow::RadialLayer, z::AbstractVecOrMat)
    return _radial_transform(first(flow.α_), first(flow.β), flow.z_0, z)
end
function _radial_transform(α_, β, z_0, z)
    α = LogExpFunctions.log1pexp(α_)            # from A.2
    β_hat = -α + LogExpFunctions.log1pexp(β)    # from A.2
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

function with_logabsdet_jacobian(flow::RadialLayer, z::AbstractVecOrMat)
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
    return (transformed, log_det_jacobian)
end

function (ib::Inverse{<:RadialLayer})(y::AbstractVector{<:Real})
    flow = ib.orig
    z0 = flow.z_0
    α = LogExpFunctions.log1pexp(first(flow.α_))            # from A.2
    α_plus_β_hat = LogExpFunctions.log1pexp(first(flow.β))  # from A.2

    # Compute the norm ``r`` from A.2.
    y_minus_z0 = y .- z0
    r = compute_r(y_minus_z0, α, α_plus_β_hat)
    γ = (α + r) / (α_plus_β_hat + r)

    return z0 .+ γ .* y_minus_z0
end

function (ib::Inverse{<:RadialLayer})(y::AbstractMatrix{<:Real})
    flow = ib.orig
    z0 = flow.z_0
    α = LogExpFunctions.log1pexp(first(flow.α_))            # from A.2
    α_plus_β_hat = LogExpFunctions.log1pexp(first(flow.β))  # from A.2

    # Compute the norm ``r`` from A.2 for each column.
    y_minus_z0 = y .- z0
    rs = mapvcat(eachcol(y_minus_z0)) do c
        return compute_r(c, α, α_plus_β_hat)
    end
    γ = reshape((α .+ rs) ./ (α_plus_β_hat .+ rs), 1, :)

    return z0 .+ γ .* y_minus_z0
end

"""
    compute_r(y_minus_z0::AbstractVector{<:Real}, α, α_plus_β_hat)

Compute the unique solution ``r`` to the equation
```math
\\|y_minus_z0\\|_2 = r \\left(1 + \\frac{α_plus_β_hat - α}{α + r}\\right)
```
subject to ``r ≥ 0`` and ``r ≠ α``.

Since ``α > 0`` and ``α_plus_β_hat > 0``, the solution is unique and given by
```math
r = (\\sqrt{(α_plus_β_hat - γ)^2 + 4 α γ} - (α_plus_β_hat - γ)) / 2,
```
where ``γ = \\|y_minus_z0\\|_2``. For details see appendix A.2 of the reference.

# References

D. Rezende, S. Mohamed (2015): Variational Inference with Normalizing Flows.
arXiv:1505.05770
"""
function compute_r(y_minus_z0::AbstractVector{<:Real}, α, α_plus_β_hat)
    γ = norm(y_minus_z0)
    a = α_plus_β_hat - γ
    r = (sqrt(a^2 + 4 * α * γ) - a) / 2
    return r
end

logabsdetjac(flow::RadialLayer, x::AbstractVecOrMat) = last(with_logabsdet_jacobian(flow, x))
