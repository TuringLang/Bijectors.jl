################################################################################
#                            Planar and Radial Flows                           #
#             Ref: Variational Inference with Normalizing Flows,               #
#               D. Rezende, S. Mohamed(2015) arXiv:1505.05770                  #
################################################################################

###############
# PlanarLayer #
###############

# TODO: add docstring

struct PlanarLayer{T1<:AbstractVector{<:Real}, T2<:Union{Real, AbstractVector{<:Real}}} <: Bijector{1}
    w::T1
    u::T1
    b::T2
end
function Base.:(==)(b1::PlanarLayer, b2::PlanarLayer)
    return b1.w == b2.w && b1.u == b2.u && b1.b == b2.b
end

function PlanarLayer(dims::Int, wrapper=identity)
    w = wrapper(randn(dims))
    u = wrapper(randn(dims))
    b = wrapper(randn(1))
    return PlanarLayer(w, u, b)
end

# all fields are numerical parameters
Functors.@functor PlanarLayer

"""
    get_u_hat(u::AbstractVector{<:Real}, w::AbstractVector{<:Real})

Return a tuple of vector ``û`` that guarantees invertibility of the planar layer, and
scalar ``wᵀ û``.

# Mathematical background

According to appendix A.1, vector ``û`` defined by
```math
û(w, u) = u + (\\log(1 + \\exp{(wᵀu)}) - 1 - wᵀu) \\frac{w}{\\|w\\|²}
```
guarantees that the planar layer ``f(z) = z + û tanh(wᵀz + b)`` is invertible for all ``w, u ∈ ℝᵈ`` and ``b ∈ ℝ``.
We can rewrite ``û`` as
```math
û = u + (\\log(1 + \\exp{(-wᵀu)}) - 1) \\frac{w}{\\|w\\|²}.
```

Additionally, we obtain
```math
wᵀû = wᵀu + \\log(1 + \\exp{(-wᵀu)}) - 1 = \\log(1 + \\exp{(wᵀu)}) - 1.
```

# References

D. Rezende, S. Mohamed (2015): Variational Inference with Normalizing Flows.
arXiv:1505.05770
"""
function get_u_hat(u::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    wT_u = dot(w, u)
    û = u .+ ((LogExpFunctions.log1pexp(-wT_u) - 1) / sum(abs2, w)) .* w
    wT_û = LogExpFunctions.log1pexp(wT_u) - 1
    return û, wT_û
end

# An internal version of the transform in eq. (10) that returns intermediate variables
function _transform(flow::PlanarLayer, z::AbstractVecOrMat{<:Real})
    w = flow.w
    b = first(flow.b)
    û, wT_û = get_u_hat(flow.u, w)
    wT_z = aT_b(w, z)
    transformed = z .+ û .* tanh.(wT_z .+ b)
    return (transformed = transformed, wT_û = wT_û, wT_z = wT_z)
end

(b::PlanarLayer)(z) = _transform(b, z).transformed

#=
Log-determinant of the Jacobian of the planar layer

The log-determinant of the Jacobian of the planar layer ``f(z) = z + û tanh(wᵀz + b)``
is given by
```math
\\log |det ∂f(z)/∂z| = \\log |1 + ûᵀsech²(wᵀz + b)w| = \\log |1 + sech²(wᵀz + b) wᵀû|.
```

Since ``0 < sech²(x) ≤ 1`` and
```math
wᵀû = wᵀu + \\log(1 + \\exp{(-wᵀu)}) - 1 = \\log(1 + \\exp{(wᵀu)}) - 1 > -1,
```
we get
```math
\\log |det ∂f(z)/∂z| = \\log(1 + sech²(wᵀz + b) wᵀû).
```
=#
function forward(flow::PlanarLayer, z::AbstractVecOrMat{<:Real})
    transformed, wT_û, wT_z = _transform(flow, z)

    # Compute ``\\log |det ∂f(z)/∂z|`` (see above).
    b = first(flow.b)
    log_det_jacobian = log1p.(wT_û .* abs2.(sech.(_vec(wT_z) .+ b)))

    return (rv = transformed, logabsdetjac = log_det_jacobian)
end

function (ib::Inverse{<:PlanarLayer})(y::AbstractVecOrMat{<:Real})
    flow = ib.orig
    w = flow.w
    b = first(flow.b)
    û, wT_û = get_u_hat(flow.u, w)

    # Find the scalar ``α`` by solving ``wᵀy = α + wᵀû tanh(α + b)``
    # (eq. (23) from appendix A.1).
    wT_y = aT_b(w, y)
    α = find_alpha.(wT_y, wT_û, b)

    # Compute ``z = y - û tanh(α + b)``.
    z = y .- û .* tanh.(α .+ b)

    return z
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
function find_alpha(wt_y::Real, wt_u_hat::Real, b::Real)
    # avoid promotions in root-finding algorithm and simplify AD dispatches
    return find_alpha(promote(wt_y, wt_u_hat, b)...)
end
function find_alpha(wt_y::T, wt_u_hat::T, b::T) where {T<:Real}
    # Compute the initial bracket (see above).
    initial_bracket = (wt_y - abs(wt_u_hat), wt_y + abs(wt_u_hat))

    # Try to solve the root-finding problem, i.e., compute a final bracket
    prob = NonlinearSolve.NonlinearProblem{false}(initial_bracket) do α, _
        α + wt_u_hat * tanh(α + b) - wt_y
    end
    sol = NonlinearSolve.solve(prob, NonlinearSolve.Falsi())
    if sol.retcode === NonlinearSolve.MAXITERS_EXCEED
        @warn "Planar layer: root finding algorithm did not converge" sol
    end

    return sol.left
end

logabsdetjac(flow::PlanarLayer, x) = forward(flow, x).logabsdetjac
isclosedform(b::Inverse{<:PlanarLayer}) = false
