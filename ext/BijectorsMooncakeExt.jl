module BijectorsMooncakeExt

using Mooncake: @is_primitive, MinimalCtx, Mooncake, CoDual, primal, tangent_type
using Bijectors: find_alpha

# Closed-form partials of `find_alpha` derived from differentiating
#   wt_y == α + wt_u_hat * tanh(α + b)
# implicitly w.r.t. each argument. `s = sech(α + b)`, `x = 1 / (1 + wt_u_hat * s^2)`.
#   ∂α/∂wt_y     = x
#   ∂α/∂wt_u_hat = -tanh(α + b) * x
#   ∂α/∂b        = x - 1
function _find_alpha_partials(wt_y::P, wt_u_hat::P, b::P) where {P<:Base.IEEEFloat}
    α = find_alpha(wt_y, wt_u_hat, b)
    s = sech(α + b)
    x = inv(1 + wt_u_hat * s^2)
    return α, x, -tanh(α + b) * x, x - one(P)
end

# Floating-point third argument.
@is_primitive(MinimalCtx, Tuple{typeof(find_alpha),P,P,P} where {P<:Base.IEEEFloat},)

function Mooncake.frule!!(
    ::Mooncake.Dual{typeof(find_alpha)},
    x::Mooncake.Dual{P},
    y::Mooncake.Dual{P},
    z::Mooncake.Dual{P},
) where {P<:Base.IEEEFloat}
    α, ∂y, ∂u, ∂b = _find_alpha_partials(
        Mooncake.primal(x), Mooncake.primal(y), Mooncake.primal(z)
    )
    dα = ∂y * Mooncake.tangent(x) + ∂u * Mooncake.tangent(y) + ∂b * Mooncake.tangent(z)
    return Mooncake.Dual(α, dα)
end

function Mooncake.rrule!!(
    ::CoDual{typeof(find_alpha)}, x::CoDual{P}, y::CoDual{P}, z::CoDual{P}
) where {P<:Base.IEEEFloat}
    α, ∂y, ∂u, ∂b = _find_alpha_partials(primal(x), primal(y), primal(z))
    find_alpha_pb(dα::P) = Mooncake.NoRData(), dα * ∂y, dα * ∂u, dα * ∂b
    return Mooncake.zero_fcodual(α), find_alpha_pb
end

# Integer-valued third argument — non-differentiable, so the corresponding partial drops.
# We verify the concrete `Integer` type does have `NoTangent` tangent, in case of unusual
# integer types.
@is_primitive(MinimalCtx, Tuple{typeof(find_alpha),P,P,Integer} where {P<:Base.IEEEFloat},)

function _assert_integer_nontangent(::Type{I}) where {I<:Integer}
    if tangent_type(I) != Mooncake.NoTangent
        throw(
            ArgumentError(
                "Integer argument has tangent type $(tangent_type(I)), should be NoTangent."
            ),
        )
    end
end

function Mooncake.frule!!(
    ::Mooncake.Dual{typeof(find_alpha)},
    x::Mooncake.Dual{P},
    y::Mooncake.Dual{P},
    z::Mooncake.Dual{I},
) where {P<:Base.IEEEFloat,I<:Integer}
    _assert_integer_nontangent(I)
    α, ∂y, ∂u, _ = _find_alpha_partials(
        Mooncake.primal(x), Mooncake.primal(y), P(Mooncake.primal(z))
    )
    dα = ∂y * Mooncake.tangent(x) + ∂u * Mooncake.tangent(y)
    return Mooncake.Dual(α, dα)
end

function Mooncake.rrule!!(
    ::CoDual{typeof(find_alpha)}, x::CoDual{P}, y::CoDual{P}, z::CoDual{I}
) where {P<:Base.IEEEFloat,I<:Integer}
    _assert_integer_nontangent(I)
    α, ∂y, ∂u, _ = _find_alpha_partials(primal(x), primal(y), P(primal(z)))
    find_alpha_pb(dα::P) = Mooncake.NoRData(), dα * ∂y, dα * ∂u, Mooncake.NoRData()
    return Mooncake.zero_fcodual(α), find_alpha_pb
end

end
