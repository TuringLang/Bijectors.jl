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

# Precompile the Mooncake reverse-mode AD pipeline for common Bijectors operations
# so that time-to-first-gradient is reduced for users. Based on the pattern from
# https://github.com/chalk-lab/Mooncake.jl/blob/main/src/precompile.jl
using PrecompileTools: @setup_workload, @compile_workload

#! format: off
@setup_workload begin
    using Bijectors:
        Bijectors, bijector, transformed, logabsdetjac, with_logabsdet_jacobian
    using Bijectors.Distributions: LogNormal, logpdf

    @compile_workload begin
        d = LogNormal()
        b = bijector(d)
        td = transformed(d)

        # Reverse-mode: differentiate logpdf of a transformed distribution
        _precompile_f(x) = logpdf(td, x)
        rule = Mooncake.build_rrule(_precompile_f, 0.5)
        Mooncake.value_and_gradient!!(rule, _precompile_f, 0.5)

        # Reverse-mode: differentiate with_logabsdet_jacobian
        _precompile_g(x) = first(with_logabsdet_jacobian(b, x))
        rule2 = Mooncake.build_rrule(_precompile_g, 1.0)
        Mooncake.value_and_gradient!!(rule2, _precompile_g, 1.0)
    end
end
#! format: on

end
