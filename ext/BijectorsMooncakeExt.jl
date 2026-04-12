module BijectorsMooncakeExt

using Mooncake:
    @is_primitive,
    MinimalCtx,
    Mooncake,
    CoDual,
    primal,
    tangent_type,
    @from_chainrules,
    prepare_pullback_cache,
    prepare_derivative_cache,
    value_and_pullback!!,
    value_and_derivative!!,
    zero_tangent
using Bijectors: find_alpha, ChainRulesCore
import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoMooncake, AutoMooncakeForward

## Reverse-mode implementations

function _value_and_gradient(f, ::AutoMooncake, x::AbstractVector{T}) where {T}
    cache = prepare_pullback_cache(f, x)
    val, (_, x_grad) = value_and_pullback!!(cache, one(T), f, x)
    return val, x_grad
end

function _value_and_jacobian(f, ::AutoMooncake, x::AbstractVector{T}) where {T}
    y = f(x)
    n_out, n_in = length(y), length(x)
    J = Matrix{T}(undef, n_out, n_in)
    cache = prepare_pullback_cache(f, x)
    dy = zeros(eltype(y), n_out)
    for i in 1:n_out
        fill!(dy, zero(eltype(y)))
        dy[i] = one(eltype(y))
        _, (_, row) = value_and_pullback!!(cache, dy, f, x)
        J[i, :] .= row
    end
    return y, J
end

## Forward-mode implementations (column-by-column JVPs)

function _value_and_gradient(f, ::AutoMooncakeForward, x::AbstractVector{T}) where {T}
    val = f(x)
    n = length(x)
    grad = Vector{T}(undef, n)
    cache = prepare_derivative_cache(f, x)
    df = zero_tangent(f)
    dx = zeros(T, n)
    for j in 1:n
        fill!(dx, zero(T))
        dx[j] = one(T)
        _, jvp = value_and_derivative!!(cache, (f, df), (x, dx))
        grad[j] = jvp
    end
    return val, grad
end

function _value_and_jacobian(f, ::AutoMooncakeForward, x::AbstractVector{T}) where {T}
    y = f(x)
    n_out, n_in = length(y), length(x)
    J = Matrix{T}(undef, n_out, n_in)
    cache = prepare_derivative_cache(f, x)
    df = zero_tangent(f)
    dx = zeros(T, n_in)
    for j in 1:n_in
        fill!(dx, zero(T))
        dx[j] = one(T)
        _, jvp = value_and_derivative!!(cache, (f, df), (x, dx))
        J[:, j] .= jvp
    end
    return y, J
end

@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float16,Float16,Float16})
@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float32,Float32,Float32})
@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float64,Float64,Float64})

# The final argument could be an Integer of some kind. This should be fine provided that
# it has tangent type equal to `NoTangent`, which means that it's non-differentiable and
# can be safely dropped. We verify that the concrete type of the Integer satisfies this
# constraint, and error if (for some reason) it does not. This should be fine unless a very
# unusual Integer type is encountered.
@is_primitive(MinimalCtx, Tuple{typeof(find_alpha),P,P,Integer} where {P<:Base.IEEEFloat})

function Mooncake.frule!!(
    ::Mooncake.Dual{typeof(find_alpha)},
    x::Mooncake.Dual{P},
    y::Mooncake.Dual{P},
    z::Mooncake.Dual{I},
) where {P<:Base.IEEEFloat,I<:Integer}
    # Require that the integer is non-differentiable.
    if tangent_type(I) != Mooncake.NoTangent
        msg = "Integer argument has tangent type $(tangent_type(I)), should be NoTangent."
        throw(ArgumentError(msg))
    end
    # Convert Mooncake.NoTangent to ChainRulesCore.NoTangent for the integer argument
    out, tangent_out = ChainRulesCore.frule(
        (
            ChainRulesCore.NoTangent(),
            Mooncake.tangent(x),
            Mooncake.tangent(y),
            ChainRulesCore.NoTangent(),
        ),
        find_alpha,
        Mooncake.primal(x),
        Mooncake.primal(y),
        Mooncake.primal(z),
    )
    return Mooncake.Dual(out, tangent_out)
end

function Mooncake.rrule!!(
    ::CoDual{typeof(find_alpha)}, x::CoDual{P}, y::CoDual{P}, z::CoDual{I}
) where {P<:Base.IEEEFloat,I<:Integer}
    # Require that the integer is non-differentiable.
    if tangent_type(I) != Mooncake.NoTangent
        msg = "Integer argument has tangent type $(tangent_type(I)), should be NoTangent."
        throw(ArgumentError(msg))
    end
    out, pb = ChainRulesCore.rrule(find_alpha, primal(x), primal(y), primal(z))
    function find_alpha_pb(dout::P)
        _, dx, dy, _ = pb(dout)
        return Mooncake.NoRData(), P(dx), P(dy), Mooncake.NoRData()
    end
    return Mooncake.zero_fcodual(out), find_alpha_pb
end

end
