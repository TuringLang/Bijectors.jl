module BijectorsMooncakeExt

using Mooncake:
    @is_primitive, MinimalCtx, Mooncake, CoDual, primal, tangent_type, @from_chainrules
using Bijectors: find_alpha, ChainRulesCore

@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float16,Float16,Float16})
@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float32,Float32,Float32})
@from_chainrules(MinimalCtx, Tuple{typeof(find_alpha),Float64,Float64,Float64})

# The final argument could be an Integer of some kind. This should be fine provided that
# it has tangent type equal to `NoTangent`, which means that it's non-differentiable and
# can be safely dropped. We verify that the concrete type of the Integer satisfies this
# constraint, and error if (for some reason) it does not. This should be fine unless a very
# unusual Integer type is encountered.
@is_primitive(MinimalCtx, Tuple{typeof(find_alpha),P,P,Integer} where {P<:Base.IEEEFloat})

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
