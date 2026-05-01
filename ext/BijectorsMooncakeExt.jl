module BijectorsMooncakeExt

using Mooncake: Mooncake
using Mooncake:
    @is_primitive,
    MinimalCtx,
    CoDual,
    tangent_type,
    @from_chainrules,
    prepare_pullback_cache,
    prepare_gradient_cache,
    prepare_derivative_cache,
    value_and_pullback!!,
    value_and_gradient!!,
    value_and_derivative!!,
    zero_tangent,
    Config,
    _copy_output,
    tangent_to_primal!!
using Bijectors: find_alpha, ChainRulesCore
import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoMooncake, AutoMooncakeForward

_mooncake_config(::Union{AutoMooncake{Nothing},AutoMooncakeForward{Nothing}}) = Config()
_mooncake_config(backend::Union{AutoMooncake,AutoMooncakeForward}) = backend.config

function _mooncake_zero_tangent_or_primal(
    x, backend::Union{AutoMooncake,AutoMooncakeForward}
)
    if _mooncake_config(backend).friendly_tangents
        return tangent_to_primal!!(_copy_output(x), zero_tangent(x))
    else
        return zero_tangent(x)
    end
end

## Reverse-mode implementations

function _value_and_gradient(f, backend::AutoMooncake, x::AbstractVector)
    cache = prepare_gradient_cache(f, x; config=_mooncake_config(backend))
    val, (_, x_grad) = value_and_gradient!!(cache, f, x)
    return val, x_grad
end

function _value_and_jacobian(f, backend::AutoMooncake, x::AbstractVector)
    cache = prepare_pullback_cache(f, x; config=_mooncake_config(backend))
    n_out, n_in = length(cache.y_cache), length(x)
    dy = zeros(eltype(cache.y_cache), n_out)
    if n_out > 0
        dy[1] = one(eltype(cache.y_cache))
    end
    val, (_, first_row) = value_and_pullback!!(cache, dy, f, x)
    if n_out == 0
        return _copy_output(val), Matrix{eltype(x)}(undef, 0, n_in)
    end
    y = _copy_output(val)
    J = Matrix{eltype(first_row)}(undef, n_out, n_in)
    J[1, :] .= first_row
    for i in 2:n_out
        fill!(dy, zero(eltype(cache.y_cache)))
        dy[i] = one(eltype(cache.y_cache))
        _, (_, row) = value_and_pullback!!(cache, dy, f, x)
        J[i, :] .= row
    end
    return y, J
end

## Forward-mode implementations (column-by-column JVPs)

function _value_and_gradient(f, backend::AutoMooncakeForward, x::AbstractVector)
    cache = prepare_gradient_cache(f, x; config=_mooncake_config(backend))
    val, (_, x_grad) = value_and_gradient!!(cache, f, x)
    return val, x_grad
end

function _value_and_jacobian(
    f, backend::AutoMooncakeForward, x::AbstractVector{T}
) where {T}
    y = f(x)
    n_out, n_in = length(y), length(x)
    cache = prepare_derivative_cache(f, x; config=_mooncake_config(backend))
    df = _mooncake_zero_tangent_or_primal(f, backend)
    if n_in == 0
        return y, Matrix{eltype(y)}(undef, n_out, 0)
    end
    dx = zeros(T, n_in)
    dx[1] = one(T)
    _, first_jvp = value_and_derivative!!(cache, (f, df), (x, dx))
    J = Matrix{eltype(first_jvp)}(undef, n_out, n_in)
    J[:, 1] .= first_jvp
    for j in 2:n_in
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
    out, pb = ChainRulesCore.rrule(
        find_alpha, Mooncake.primal(x), Mooncake.primal(y), Mooncake.primal(z)
    )
    function find_alpha_pb(dout::P)
        _, dx, dy, _ = pb(dout)
        return Mooncake.NoRData(), P(dx), P(dy), Mooncake.NoRData()
    end
    return Mooncake.zero_fcodual(out), find_alpha_pb
end

end
