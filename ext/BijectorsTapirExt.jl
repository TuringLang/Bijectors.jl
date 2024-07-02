module BijectorsTapirExt

if isdefined(Base, :get_extension)
    using Tapir: @is_primitive, MinimalCtx, Tapir, CoDual, primal, tangent_type, @from_rrule
    using Bijectors: find_alpha
    using ChainRulesCore: rrule
else
    using ..Tapir: @is_primitive, MinimalCtx, Tapir, primal, tangent_type, @from_rrule
    using ..Bijectors: find_alpha, rrule
    using ..ChainRulesCore: rrule
end

for P in [Float16, Float32, Float64]
    @from_rrule(MinimalCtx, Tuple{typeof(find_alpha),P,P,P})
end

# The final argument could be an Integer of some kind. This should be fine provided that
# it has tangent type equal to `NoTangent`, which means that it's non-differentiable and
# can be safely dropped. We verify that the concrete type of the Integer satisfies this
# constraint, and error if (for some reason) it does not. This should be fine unless a very
# unusual Integer type is encountered.
@is_primitive(MinimalCtx, Tuple{typeof(find_alpha),P,P,Integer} where {P<:Base.IEEEFloat})

function Tapir.rrule!!(
    ::CoDual{typeof(find_alpha)}, x::CoDual{P}, y::CoDual{P}, z::CoDual{I}
) where {P<:Base.IEEEFloat,I<:Integer}
    # Require that the integer is non-differentiable.
    if tangent_type(I) != Tapir.NoTangent
        msg = "Integer argument has tangent type $(tangent_type(I)), should be NoTangent."
        throw(ArgumentError(msg))
    end
    out, pb = rrule(find_alpha, primal(x), primal(y), primal(z))
    function find_alpha_pb(dout::P)
        _, dx, dy, _ = pb(dout)
        return Tapir.NoRData(), P(dx), P(dy), Tapir.NoRData()
    end
    return Tapir.zero_fcodual(out), find_alpha_pb
end

end
