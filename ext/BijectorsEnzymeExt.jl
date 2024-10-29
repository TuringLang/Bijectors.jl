module BijectorsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme: Enzyme
    using EnzymeCore: EnzymeCore
    using Bijectors: Bijectors, ChainRulesCore
else
    using ..Enzyme: Enzyme
    using ..EnzymeCore: EnzymeCore
    using ..Bijectors: Bijectors, ChainRulesCore
end

#= NOTE(penelopeysm):
Changes made to the way extensions were loaded in Julia 1.11.1 mean that it
is no longer sufficient to call Enzyme.@import_rrule and
Enzyme.@import_frule, as we did in previous versions. This is because both of
those macros rely on a method which is defined in EnzymeChainRulesCoreExt,
and on 1.11.1+, that extension is _not_ loaded before BijectorsEnzymeExt is
loaded. (In the past, for reasons which are not fully clear,
EnzymeChainRulesCoreExt _does_ get loaded first.)

See https://github.com/TuringLang/Bijectors.jl/pull/333 and 
https://github.com/TuringLang/Bijectors.jl/pull/337 for further context about
the underlying issue and the other ways that were explored to resolve it.

However, on versions of Julia where the 'default' extension resolution occurs,
we can still use the macros (see the else clause below). We do this to ensure
that the code is compatible with what may potentially be different versions of
Enzyme.

The code in the if clause was derived by calling @macroexpand on @import_rrule
and @import_frule, then replacing `$(Expr(:meta, :inline))` with
`Base.@_inline_meta`.

Note that this was done using Enzyme v0.12.36. This code will fail to track any
upstream changes to EnzymeChainRulesCoreExt, so there is no guarantee that this
code will work with later versions of Enzyme.
=#
@static if v"1.11.1" <= VERSION < v"1.12"
    function (Enzyme.EnzymeRules).augmented_primal(
        var"#238#config",
        var"#239#fn"::var"#246#FA",
        ::Enzyme.Type{var"#245#RetAnnotation"},
        var"#241#arg_1"::var"#247#AN_1",
        var"#242#arg_2"::var"#248#AN_2",
        var"#243#arg_3"::var"#249#AN_3";
        var"#244#kwargs"...,
    ) where {
        var"#245#RetAnnotation",
        var"#246#FA"<:Enzyme.Annotation{<:typeof(Bijectors.find_alpha)},
        var"#247#AN_1"<:Enzyme.Annotation{<:Real},
        var"#248#AN_2"<:Enzyme.Annotation{<:Real},
        var"#249#AN_3"<:Enzyme.Annotation{<:Real},
    }
        var"#231#primcopy_1" =
            if ((EnzymeCore.EnzymeRules.overwritten)(var"#238#config"))[1 + 1]
                Enzyme.deepcopy((var"#241#arg_1").val)
            else
                (var"#241#arg_1").val
            end
        var"#232#primcopy_2" =
            if ((EnzymeCore.EnzymeRules.overwritten)(var"#238#config"))[2 + 1]
                Enzyme.deepcopy((var"#242#arg_2").val)
            else
                (var"#242#arg_2").val
            end
        var"#233#primcopy_3" =
            if ((EnzymeCore.EnzymeRules.overwritten)(var"#238#config"))[3 + 1]
                Enzyme.deepcopy((var"#243#arg_3").val)
            else
                (var"#243#arg_3").val
            end
        (var"#234#res", var"#235#pullback") = if var"#245#RetAnnotation" <: Enzyme.Const
            (
                (var"#239#fn").val(
                    var"#231#primcopy_1",
                    var"#232#primcopy_2",
                    var"#233#primcopy_3";
                    var"#244#kwargs"...,
                ),
                Enzyme.nothing,
            )
        else
            (ChainRulesCore).rrule(
                (var"#239#fn").val,
                var"#231#primcopy_1",
                var"#232#primcopy_2",
                var"#233#primcopy_3";
                var"#244#kwargs"...,
            )
        end
        var"#236#primal" = if (Enzyme.EnzymeRules).needs_primal(var"#238#config")
            var"#234#res"
        else
            Enzyme.nothing
        end
        var"#237#shadow" = if !((Enzyme.EnzymeRules).needs_shadow(var"#238#config"))
            Enzyme.nothing
        else
            if (Enzyme.EnzymeRules).width(var"#238#config") == 1
                (Enzyme.Enzyme).make_zero(var"#234#res")
            else
                Enzyme.ntuple(
                    Enzyme.Val((Enzyme.EnzymeRules).width(var"#238#config"))
                ) do var"#250#j"
                    Base.@_inline_meta
                    (Enzyme.Enzyme).make_zero(var"#234#res")
                end
            end
        end
        return (Enzyme.EnzymeRules).AugmentedReturn(
            var"#236#primal", var"#237#shadow", (var"#237#shadow", var"#235#pullback")
        )
    end

    function (Enzyme.EnzymeRules).reverse(
        var"#254#config",
        var"#255#fn"::var"#264#FA",
        ::Enzyme.Type{var"#262#RetAnnotation"},
        var"#257#tape"::var"#263#TapeTy",
        var"#258#arg_1"::var"#265#AN_1",
        var"#259#arg_2"::var"#266#AN_2",
        var"#260#arg_3"::var"#267#AN_3";
        var"#261#kwargs"...,
    ) where {
        var"#262#RetAnnotation",
        var"#263#TapeTy",
        var"#264#FA"<:Enzyme.Annotation{<:typeof(Bijectors.find_alpha)},
        var"#265#AN_1"<:Enzyme.Annotation{<:Real},
        var"#266#AN_2"<:Enzyme.Annotation{<:Real},
        var"#267#AN_3"<:Enzyme.Annotation{<:Real},
    }
        if !(var"#262#RetAnnotation" <: Enzyme.Const)
            (var"#251#shadow", var"#252#pullback") = var"#257#tape"
            var"#253#tcomb" = Enzyme.ntuple(
                Enzyme.Val((Enzyme.EnzymeRules).width(var"#254#config"))
            ) do var"#272#batch_i"
                Base.@_inline_meta
                var"#268#shad" = if (Enzyme.EnzymeRules).width(var"#254#config") == 1
                    var"#251#shadow"
                else
                    var"#251#shadow"[var"#272#batch_i"]
                end
                var"#269#res" = var"#252#pullback"(var"#268#shad")
                for (var"#270#cr", var"#271#en") in Enzyme.zip(
                    var"#269#res",
                    (var"#255#fn", var"#258#arg_1", var"#259#arg_2", var"#260#arg_3"),
                )
                    if var"#271#en" isa Enzyme.Const ||
                        var"#270#cr" isa (ChainRulesCore).NoTangent
                        continue
                    end
                    if var"#271#en" isa Enzyme.Active
                        continue
                    end
                    if (Enzyme.EnzymeRules).width(var"#254#config") == 1
                        (var"#271#en").dval .+= var"#270#cr"
                    else
                        (var"#271#en").dval[var"#272#batch_i"] .+= var"#270#cr"
                    end
                end
                (
                    if var"#255#fn" isa Enzyme.Active
                        var"#269#res"[1]
                    else
                        Enzyme.nothing
                    end,
                    if var"#258#arg_1" isa Enzyme.Active
                        if var"#269#res"[1 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#258#arg_1")
                        else
                            (ChainRulesCore).unthunk(var"#269#res"[1 + 1])
                        end
                    else
                        Enzyme.nothing
                    end,
                    if var"#259#arg_2" isa Enzyme.Active
                        if var"#269#res"[2 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#259#arg_2")
                        else
                            (ChainRulesCore).unthunk(var"#269#res"[2 + 1])
                        end
                    else
                        Enzyme.nothing
                    end,
                    if var"#260#arg_3" isa Enzyme.Active
                        if var"#269#res"[3 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#260#arg_3")
                        else
                            (ChainRulesCore).unthunk(var"#269#res"[3 + 1])
                        end
                    else
                        Enzyme.nothing
                    end,
                )
            end
            return (
                begin
                    if var"#258#arg_1" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#254#config") == 1
                            (var"#253#tcomb"[1])[1 + 1]
                        else
                            Enzyme.ntuple(
                                Enzyme.Val((Enzyme.EnzymeRules).width(var"#254#config"))
                            ) do var"#273#batch_i"
                                Base.@_inline_meta
                                (var"#253#tcomb"[var"#273#batch_i"])[1 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end,
                begin
                    if var"#259#arg_2" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#254#config") == 1
                            (var"#253#tcomb"[1])[2 + 1]
                        else
                            Enzyme.ntuple(
                                Enzyme.Val((Enzyme.EnzymeRules).width(var"#254#config"))
                            ) do var"#274#batch_i"
                                Base.@_inline_meta
                                (var"#253#tcomb"[var"#274#batch_i"])[2 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end,
                begin
                    if var"#260#arg_3" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#254#config") == 1
                            (var"#253#tcomb"[1])[3 + 1]
                        else
                            Enzyme.ntuple(
                                Enzyme.Val((Enzyme.EnzymeRules).width(var"#254#config"))
                            ) do var"#275#batch_i"
                                Base.@_inline_meta
                                (var"#253#tcomb"[var"#275#batch_i"])[3 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end,
            )
        end
        return (Enzyme.nothing, Enzyme.nothing, Enzyme.nothing)
    end

    function (Enzyme.EnzymeRules).reverse(
        var"#280#config",
        var"#281#fn"::var"#290#FA",
        var"#282#dval"::Enzyme.Active{var"#288#RetAnnotation"},
        var"#283#tape"::var"#289#TapeTy",
        var"#284#arg_1"::var"#291#AN_1",
        var"#285#arg_2"::var"#292#AN_2",
        var"#286#arg_3"::var"#293#AN_3";
        var"#287#kwargs"...,
    ) where {
        var"#288#RetAnnotation",
        var"#289#TapeTy",
        var"#290#FA"<:Enzyme.Annotation{<:typeof(Bijectors.find_alpha)},
        var"#291#AN_1"<:Enzyme.Annotation{<:Real},
        var"#292#AN_2"<:Enzyme.Annotation{<:Real},
        var"#293#AN_3"<:Enzyme.Annotation{<:Real},
    }
        (var"#276#oldshadow", var"#277#pullback") = var"#283#tape"
        var"#278#shadow" = (var"#282#dval").val
        var"#279#tcomb" = Enzyme.ntuple(
            Enzyme.Val((Enzyme.EnzymeRules).width(var"#280#config"))
        ) do var"#298#batch_i"
            Base.@_inline_meta
            var"#294#shad" = if (Enzyme.EnzymeRules).width(var"#280#config") == 1
                var"#278#shadow"
            else
                var"#278#shadow"[var"#298#batch_i"]
            end
            var"#295#res" = var"#277#pullback"(var"#294#shad")
            for (var"#296#cr", var"#297#en") in Enzyme.zip(
                var"#295#res",
                (var"#281#fn", var"#284#arg_1", var"#285#arg_2", var"#286#arg_3"),
            )
                if var"#297#en" isa Enzyme.Const || var"#296#cr" isa (ChainRulesCore).NoTangent
                    continue
                end
                if var"#297#en" isa Enzyme.Active
                    continue
                end
                if (Enzyme.EnzymeRules).width(var"#280#config") == 1
                    (var"#297#en").dval .+= var"#296#cr"
                else
                    (var"#297#en").dval[var"#298#batch_i"] .+= var"#296#cr"
                end
            end
            (
                if var"#281#fn" isa Enzyme.Active
                    var"#295#res"[1]
                else
                    Enzyme.nothing
                end,
                if var"#284#arg_1" isa Enzyme.Active
                    if var"#295#res"[1 + 1] isa (ChainRulesCore).NoTangent
                        Enzyme.zero(var"#284#arg_1")
                    else
                        (ChainRulesCore).unthunk(var"#295#res"[1 + 1])
                    end
                else
                    Enzyme.nothing
                end,
                if var"#285#arg_2" isa Enzyme.Active
                    if var"#295#res"[2 + 1] isa (ChainRulesCore).NoTangent
                        Enzyme.zero(var"#285#arg_2")
                    else
                        (ChainRulesCore).unthunk(var"#295#res"[2 + 1])
                    end
                else
                    Enzyme.nothing
                end,
                if var"#286#arg_3" isa Enzyme.Active
                    if var"#295#res"[3 + 1] isa (ChainRulesCore).NoTangent
                        Enzyme.zero(var"#286#arg_3")
                    else
                        (ChainRulesCore).unthunk(var"#295#res"[3 + 1])
                    end
                else
                    Enzyme.nothing
                end,
            )
        end
        return (
            begin
                if var"#284#arg_1" isa Enzyme.Active
                    if (Enzyme.EnzymeRules).width(var"#280#config") == 1
                        (var"#279#tcomb"[1])[1 + 1]
                    else
                        Enzyme.ntuple(
                            Enzyme.Val((Enzyme.EnzymeRules).width(var"#280#config"))
                        ) do var"#299#batch_i"
                            Base.@_inline_meta
                            (var"#279#tcomb"[var"#299#batch_i"])[1 + 1]
                        end
                    end
                else
                    Enzyme.nothing
                end
            end,
            begin
                if var"#285#arg_2" isa Enzyme.Active
                    if (Enzyme.EnzymeRules).width(var"#280#config") == 1
                        (var"#279#tcomb"[1])[2 + 1]
                    else
                        Enzyme.ntuple(
                            Enzyme.Val((Enzyme.EnzymeRules).width(var"#280#config"))
                        ) do var"#300#batch_i"
                            Base.@_inline_meta
                            (var"#279#tcomb"[var"#300#batch_i"])[2 + 1]
                        end
                    end
                else
                    Enzyme.nothing
                end
            end,
            begin
                if var"#286#arg_3" isa Enzyme.Active
                    if (Enzyme.EnzymeRules).width(var"#280#config") == 1
                        (var"#279#tcomb"[1])[3 + 1]
                    else
                        Enzyme.ntuple(
                            Enzyme.Val((Enzyme.EnzymeRules).width(var"#280#config"))
                        ) do var"#301#batch_i"
                            Base.@_inline_meta
                            (var"#279#tcomb"[var"#301#batch_i"])[3 + 1]
                        end
                    end
                else
                    Enzyme.nothing
                end
            end,
        )
    end

    function (Enzyme.EnzymeRules).forward(
        var"#308#fn"::var"#315#FA",
        ::Enzyme.Type{var"#314#RetAnnotation"},
        var"#310#arg_1"::var"#316#AN_1",
        var"#311#arg_2"::var"#317#AN_2",
        var"#312#arg_3"::var"#318#AN_3";
        var"#313#kwargs"...,
    ) where {
        var"#314#RetAnnotation",
        var"#315#FA"<:Enzyme.Annotation{<:typeof(Bijectors.find_alpha)},
        var"#316#AN_1"<:Enzyme.Annotation{<:Real},
        var"#317#AN_2"<:Enzyme.Annotation{<:Real},
        var"#318#AN_3"<:Enzyme.Annotation{<:Real},
    }
        var"#302#batchsize" = Enzyme.same_or_one(
            1, var"#310#arg_1", var"#311#arg_2", var"#312#arg_3"
        )
        if var"#302#batchsize" == 1
            var"#306#dfn" = if var"#308#fn" isa Enzyme.Const
                (ChainRulesCore).NoTangent()
            else
                (var"#308#fn").dval
            end
            var"#303#cres" = (ChainRulesCore).frule(
                (
                    var"#306#dfn",
                    if var"#310#arg_1" isa Enzyme.Const
                        (ChainRulesCore).NoTangent()
                    else
                        (var"#310#arg_1").dval
                    end,
                    if var"#311#arg_2" isa Enzyme.Const
                        (ChainRulesCore).NoTangent()
                    else
                        (var"#311#arg_2").dval
                    end,
                    if var"#312#arg_3" isa Enzyme.Const
                        (ChainRulesCore).NoTangent()
                    else
                        (var"#312#arg_3").dval
                    end,
                ),
                (var"#308#fn").val,
                (var"#310#arg_1").val,
                (var"#311#arg_2").val,
                (var"#312#arg_3").val;
                var"#313#kwargs"...,
            )
            if var"#314#RetAnnotation" <: Enzyme.Const
                return var"#303#cres"[2]::Enzyme.eltype(var"#314#RetAnnotation")
            elseif var"#314#RetAnnotation" <: Enzyme.Duplicated
                return Enzyme.Duplicated(var"#303#cres"[1], var"#303#cres"[2])
            elseif var"#314#RetAnnotation" <: Enzyme.DuplicatedNoNeed
                return var"#303#cres"[2]::Enzyme.eltype(var"#314#RetAnnotation")
            else
                if false
                    nothing
                else
                    Base.throw(Base.AssertionError("false"))
                end
            end
        else
            if var"#314#RetAnnotation" <: Enzyme.Const
                var"#303#cres" =
                    Enzyme.ntuple(Enzyme.Val(var"#302#batchsize")) do var"#305#i"
                        Base.@_inline_meta
                        var"#306#dfn" = if var"#308#fn" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#308#fn").dval[var"#305#i"]
                        end
                        (ChainRulesCore).frule(
                            (
                                var"#306#dfn",
                                if var"#310#arg_1" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#310#arg_1").dval[var"#305#i"]
                                end,
                                if var"#311#arg_2" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#311#arg_2").dval[var"#305#i"]
                                end,
                                if var"#312#arg_3" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#312#arg_3").dval[var"#305#i"]
                                end,
                            ),
                            (var"#308#fn").val,
                            (var"#310#arg_1").val,
                            (var"#311#arg_2").val,
                            (var"#312#arg_3").val;
                            var"#313#kwargs"...,
                        )
                    end
                return (var"#303#cres"[1])[2]::Enzyme.eltype(var"#314#RetAnnotation")
            elseif var"#314#RetAnnotation" <: Enzyme.BatchDuplicated
                var"#304#cres1" = begin
                    var"#305#i" = 1
                    var"#306#dfn" = if var"#308#fn" isa Enzyme.Const
                        (ChainRulesCore).NoTangent()
                    else
                        (var"#308#fn").dval[var"#305#i"]
                    end
                    (ChainRulesCore).frule(
                        (
                            var"#306#dfn",
                            if var"#310#arg_1" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#310#arg_1").dval[var"#305#i"]
                            end,
                            if var"#311#arg_2" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#311#arg_2").dval[var"#305#i"]
                            end,
                            if var"#312#arg_3" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#312#arg_3").dval[var"#305#i"]
                            end,
                        ),
                        (var"#308#fn").val,
                        (var"#310#arg_1").val,
                        (var"#311#arg_2").val,
                        (var"#312#arg_3").val;
                        var"#313#kwargs"...,
                    )
                end
                var"#307#batches" =
                    Enzyme.ntuple(Enzyme.Val(var"#302#batchsize" - 1)) do var"#323#j"
                        Base.@_inline_meta
                        var"#305#i" = var"#323#j" + 1
                        var"#306#dfn" = if var"#308#fn" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#308#fn").dval[var"#305#i"]
                        end
                        ((ChainRulesCore).frule(
                            (
                                var"#306#dfn",
                                if var"#310#arg_1" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#310#arg_1").dval[var"#305#i"]
                                end,
                                if var"#311#arg_2" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#311#arg_2").dval[var"#305#i"]
                                end,
                                if var"#312#arg_3" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#312#arg_3").dval[var"#305#i"]
                                end,
                            ),
                            (var"#308#fn").val,
                            (var"#310#arg_1").val,
                            (var"#311#arg_2").val,
                            (var"#312#arg_3").val;
                            var"#313#kwargs"...,
                        ))[2]
                    end
                return Enzyme.BatchDuplicated(
                    var"#304#cres1"[1], (var"#304#cres1"[2], var"#307#batches"...)
                )
            elseif var"#314#RetAnnotation" <: Enzyme.BatchDuplicatedNoNeed
                Enzyme.ntuple(Enzyme.Val(var"#302#batchsize")) do var"#305#i"
                    Base.@_inline_meta
                    var"#306#dfn" = if var"#308#fn" isa Enzyme.Const
                        (ChainRulesCore).NoTangent()
                    else
                        (var"#308#fn").dval[var"#305#i"]
                    end
                    ((ChainRulesCore).frule(
                        (
                            var"#306#dfn",
                            if var"#310#arg_1" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#310#arg_1").dval[var"#305#i"]
                            end,
                            if var"#311#arg_2" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#311#arg_2").dval[var"#305#i"]
                            end,
                            if var"#312#arg_3" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#312#arg_3").dval[var"#305#i"]
                            end,
                        ),
                        (var"#308#fn").val,
                        (var"#310#arg_1").val,
                        (var"#311#arg_2").val,
                        (var"#312#arg_3").val;
                        var"#313#kwargs"...,
                    ))[2]
                end
            else
                if false
                    nothing
                else
                    Base.throw(Base.AssertionError("false"))
                end
            end
        end
    end
else
    Enzyme.@import_rrule typeof(Bijectors.find_alpha) Real Real Real
    Enzyme.@import_frule typeof(Bijectors.find_alpha) Real Real Real
end

end  # module
