module BijectorsEnzymeExt

using Enzyme: @import_rrule, @import_frule, Enzyme, EnzymeCore
using Bijectors: find_alpha, ChainRulesCore

@static if VERSION == v"1.11.1"
    # @import_rrule
    function (Enzyme.EnzymeRules).augmented_primal(var"#8#config", var"#9#fn"::var"#16#FA", ::Enzyme.Type{var"#15#RetAnnotation"}, var"#11#arg_1"::var"#17#AN_1", var"#12#arg_2"::var"#18#AN_2", var"#13#arg_3"::var"#19#AN_3"; var"#14#kwargs"...) where {var"#15#RetAnnotation", var"#16#FA" <: Enzyme.Annotation{<:typeof(find_alpha)}, var"#17#AN_1" <: Enzyme.Annotation{<:Real}, var"#18#AN_2" <: Enzyme.Annotation{<:Real}, var"#19#AN_3" <: Enzyme.Annotation{<:Real}}
        var"#1#primcopy_1" = if ((EnzymeCore.EnzymeRules.overwritten)(var"#8#config"))[1 + 1]
                Enzyme.deepcopy((var"#11#arg_1").val)
            else
                (var"#11#arg_1").val
            end
        var"#2#primcopy_2" = if ((EnzymeCore.EnzymeRules.overwritten)(var"#8#config"))[2 + 1]
                Enzyme.deepcopy((var"#12#arg_2").val)
            else
                (var"#12#arg_2").val
            end
        var"#3#primcopy_3" = if ((EnzymeCore.EnzymeRules.overwritten)(var"#8#config"))[3 + 1]
                Enzyme.deepcopy((var"#13#arg_3").val)
            else
                (var"#13#arg_3").val
            end
        (var"#4#res", var"#5#pullback") = if var"#15#RetAnnotation" <: Enzyme.Const
                ((var"#9#fn").val(var"#1#primcopy_1", var"#2#primcopy_2", var"#3#primcopy_3"; var"#14#kwargs"...), Enzyme.nothing)
            else
                (ChainRulesCore).rrule((var"#9#fn").val, var"#1#primcopy_1", var"#2#primcopy_2", var"#3#primcopy_3"; var"#14#kwargs"...)
            end
        var"#6#primal" = if (Enzyme.EnzymeRules).needs_primal(var"#8#config")
                var"#4#res"
            else
                Enzyme.nothing
            end
        var"#7#shadow" = if !((Enzyme.EnzymeRules).needs_shadow(var"#8#config"))
                Enzyme.nothing
            else
                if (Enzyme.EnzymeRules).width(var"#8#config") == 1
                    (Enzyme.Enzyme).make_zero(var"#4#res")
                else
                    Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#8#config"))) do var"#20#j"
                        Base.@_inline_meta
                        (Enzyme.Enzyme).make_zero(var"#4#res")
                    end
                end
            end
        return (Enzyme.EnzymeRules).AugmentedReturn(var"#6#primal", var"#7#shadow", (var"#7#shadow", var"#5#pullback"))
    end
    function (Enzyme.EnzymeRules).reverse(var"#24#config", var"#25#fn"::var"#34#FA", ::Enzyme.Type{var"#32#RetAnnotation"}, var"#27#tape"::var"#33#TapeTy", var"#28#arg_1"::var"#35#AN_1", var"#29#arg_2"::var"#36#AN_2", var"#30#arg_3"::var"#37#AN_3"; var"#31#kwargs"...) where {var"#32#RetAnnotation", var"#33#TapeTy", var"#34#FA" <: Enzyme.Annotation{<:typeof(find_alpha)}, var"#35#AN_1" <: Enzyme.Annotation{<:Real}, var"#36#AN_2" <: Enzyme.Annotation{<:Real}, var"#37#AN_3" <: Enzyme.Annotation{<:Real}}
        if !(var"#32#RetAnnotation" <: Enzyme.Const)
            (var"#21#shadow", var"#22#pullback") = var"#27#tape"
            var"#23#tcomb" = Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#24#config"))) do var"#42#batch_i"
                    Base.@_inline_meta
                    var"#38#shad" = if (Enzyme.EnzymeRules).width(var"#24#config") == 1
                            var"#21#shadow"
                        else
                            var"#21#shadow"[var"#42#batch_i"]
                        end
                    var"#39#res" = var"#22#pullback"(var"#38#shad")
                    for (var"#40#cr", var"#41#en") = Enzyme.zip(var"#39#res", (var"#25#fn", var"#28#arg_1", var"#29#arg_2", var"#30#arg_3"))
                        if var"#41#en" isa Enzyme.Const || var"#40#cr" isa (ChainRulesCore).NoTangent
                            continue
                        end
                        if var"#41#en" isa Enzyme.Active
                            continue
                        end
                        if (Enzyme.EnzymeRules).width(var"#24#config") == 1
                            (var"#41#en").dval .+= var"#40#cr"
                        else
                            (var"#41#en").dval[var"#42#batch_i"] .+= var"#40#cr"
                        end
                    end
                    (if var"#25#fn" isa Enzyme.Active
                            var"#39#res"[1]
                        else
                            Enzyme.nothing
                        end, if var"#28#arg_1" isa Enzyme.Active
                            if var"#39#res"[1 + 1] isa (ChainRulesCore).NoTangent
                                Enzyme.zero(var"#28#arg_1")
                            else
                                (ChainRulesCore).unthunk(var"#39#res"[1 + 1])
                            end
                        else
                            Enzyme.nothing
                        end, if var"#29#arg_2" isa Enzyme.Active
                            if var"#39#res"[2 + 1] isa (ChainRulesCore).NoTangent
                                Enzyme.zero(var"#29#arg_2")
                            else
                                (ChainRulesCore).unthunk(var"#39#res"[2 + 1])
                            end
                        else
                            Enzyme.nothing
                        end, if var"#30#arg_3" isa Enzyme.Active
                            if var"#39#res"[3 + 1] isa (ChainRulesCore).NoTangent
                                Enzyme.zero(var"#30#arg_3")
                            else
                                (ChainRulesCore).unthunk(var"#39#res"[3 + 1])
                            end
                        else
                            Enzyme.nothing
                        end)
                end
            return (begin
                        if var"#28#arg_1" isa Enzyme.Active
                            if (Enzyme.EnzymeRules).width(var"#24#config") == 1
                                (var"#23#tcomb"[1])[1 + 1]
                            else
                                Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#24#config"))) do var"#43#batch_i"
                                    Base.@_inline_meta
                                    (var"#23#tcomb"[var"#43#batch_i"])[1 + 1]
                                end
                            end
                        else
                            Enzyme.nothing
                        end
                    end, begin
                        if var"#29#arg_2" isa Enzyme.Active
                            if (Enzyme.EnzymeRules).width(var"#24#config") == 1
                                (var"#23#tcomb"[1])[2 + 1]
                            else
                                Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#24#config"))) do var"#44#batch_i"
                                    Base.@_inline_meta
                                    (var"#23#tcomb"[var"#44#batch_i"])[2 + 1]
                                end
                            end
                        else
                            Enzyme.nothing
                        end
                    end, begin
                        if var"#30#arg_3" isa Enzyme.Active
                            if (Enzyme.EnzymeRules).width(var"#24#config") == 1
                                (var"#23#tcomb"[1])[3 + 1]
                            else
                                Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#24#config"))) do var"#45#batch_i"
                                    Base.@_inline_meta
                                    (var"#23#tcomb"[var"#45#batch_i"])[3 + 1]
                                end
                            end
                        else
                            Enzyme.nothing
                        end
                    end)
        end
        return (Enzyme.nothing, Enzyme.nothing, Enzyme.nothing)
    end
    function (Enzyme.EnzymeRules).reverse(var"#50#config", var"#51#fn"::var"#60#FA", var"#52#dval"::Enzyme.Active{var"#58#RetAnnotation"}, var"#53#tape"::var"#59#TapeTy", var"#54#arg_1"::var"#61#AN_1", var"#55#arg_2"::var"#62#AN_2", var"#56#arg_3"::var"#63#AN_3"; var"#57#kwargs"...) where {var"#58#RetAnnotation", var"#59#TapeTy", var"#60#FA" <: Enzyme.Annotation{<:typeof(find_alpha)}, var"#61#AN_1" <: Enzyme.Annotation{<:Real}, var"#62#AN_2" <: Enzyme.Annotation{<:Real}, var"#63#AN_3" <: Enzyme.Annotation{<:Real}}
        (var"#46#oldshadow", var"#47#pullback") = var"#53#tape"
        var"#48#shadow" = (var"#52#dval").val
        var"#49#tcomb" = Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#50#config"))) do var"#68#batch_i"
                Base.@_inline_meta
                var"#64#shad" = if (Enzyme.EnzymeRules).width(var"#50#config") == 1
                        var"#48#shadow"
                    else
                        var"#48#shadow"[var"#68#batch_i"]
                    end
                var"#65#res" = var"#47#pullback"(var"#64#shad")
                for (var"#66#cr", var"#67#en") = Enzyme.zip(var"#65#res", (var"#51#fn", var"#54#arg_1", var"#55#arg_2", var"#56#arg_3"))
                    if var"#67#en" isa Enzyme.Const || var"#66#cr" isa (ChainRulesCore).NoTangent
                        continue
                    end
                    if var"#67#en" isa Enzyme.Active
                        continue
                    end
                    if (Enzyme.EnzymeRules).width(var"#50#config") == 1
                        (var"#67#en").dval .+= var"#66#cr"
                    else
                        (var"#67#en").dval[var"#68#batch_i"] .+= var"#66#cr"
                    end
                end
                (if var"#51#fn" isa Enzyme.Active
                        var"#65#res"[1]
                    else
                        Enzyme.nothing
                    end, if var"#54#arg_1" isa Enzyme.Active
                        if var"#65#res"[1 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#54#arg_1")
                        else
                            (ChainRulesCore).unthunk(var"#65#res"[1 + 1])
                        end
                    else
                        Enzyme.nothing
                    end, if var"#55#arg_2" isa Enzyme.Active
                        if var"#65#res"[2 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#55#arg_2")
                        else
                            (ChainRulesCore).unthunk(var"#65#res"[2 + 1])
                        end
                    else
                        Enzyme.nothing
                    end, if var"#56#arg_3" isa Enzyme.Active
                        if var"#65#res"[3 + 1] isa (ChainRulesCore).NoTangent
                            Enzyme.zero(var"#56#arg_3")
                        else
                            (ChainRulesCore).unthunk(var"#65#res"[3 + 1])
                        end
                    else
                        Enzyme.nothing
                    end)
            end
        return (begin
                    if var"#54#arg_1" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#50#config") == 1
                            (var"#49#tcomb"[1])[1 + 1]
                        else
                            Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#50#config"))) do var"#69#batch_i"
                                Base.@_inline_meta
                                (var"#49#tcomb"[var"#69#batch_i"])[1 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end, begin
                    if var"#55#arg_2" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#50#config") == 1
                            (var"#49#tcomb"[1])[2 + 1]
                        else
                            Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#50#config"))) do var"#70#batch_i"
                                Base.@_inline_meta
                                (var"#49#tcomb"[var"#70#batch_i"])[2 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end, begin
                    if var"#56#arg_3" isa Enzyme.Active
                        if (Enzyme.EnzymeRules).width(var"#50#config") == 1
                            (var"#49#tcomb"[1])[3 + 1]
                        else
                            Enzyme.ntuple(Enzyme.Val((Enzyme.EnzymeRules).width(var"#50#config"))) do var"#71#batch_i"
                                Base.@_inline_meta
                                (var"#49#tcomb"[var"#71#batch_i"])[3 + 1]
                            end
                        end
                    else
                        Enzyme.nothing
                    end
                end)
    end

    # @import_frule
    function (Enzyme.EnzymeRules).forward(var"#78#config", var"#79#fn"::var"#86#FA", ::Enzyme.Type{var"#85#RetAnnotation"}, var"#81#arg_1"::var"#87#AN_1", var"#82#arg_2"::var"#88#AN_2", var"#83#arg_3"::var"#89#AN_3"; var"#84#kwargs"...) where {var"#85#RetAnnotation", var"#86#FA" <: Enzyme.Annotation{<:typeof(find_alpha)}, var"#87#AN_1" <: Enzyme.Annotation{<:Real}, var"#88#AN_2" <: Enzyme.Annotation{<:Real}, var"#89#AN_3" <: Enzyme.Annotation{<:Real}}
        var"#72#batchsize" = Enzyme.same_or_one(1, var"#81#arg_1", var"#82#arg_2", var"#83#arg_3")
        if var"#72#batchsize" == 1
            var"#76#dfn" = if var"#79#fn" isa Enzyme.Const
                    (ChainRulesCore).NoTangent()
                else
                    (var"#79#fn").dval
                end
            var"#73#cres" = (ChainRulesCore).frule((var"#76#dfn", if var"#81#arg_1" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#81#arg_1").dval
                        end, if var"#82#arg_2" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#82#arg_2").dval
                        end, if var"#83#arg_3" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#83#arg_3").dval
                        end), (var"#79#fn").val, (var"#81#arg_1").val, (var"#82#arg_2").val, (var"#83#arg_3").val; var"#84#kwargs"...)
            if var"#85#RetAnnotation" <: Enzyme.Const
                return var"#73#cres"[2]::Enzyme.eltype(var"#85#RetAnnotation")
            elseif #= /Users/pyong/.julia/packages/Enzyme/VSRgT/ext/EnzymeChainRulesCoreExt.jl:64 =# var"#85#RetAnnotation" <: Enzyme.Duplicated
                return Enzyme.Duplicated(var"#73#cres"[1], var"#73#cres"[2])
            elseif #= /Users/pyong/.julia/packages/Enzyme/VSRgT/ext/EnzymeChainRulesCoreExt.jl:66 =# var"#85#RetAnnotation" <: Enzyme.DuplicatedNoNeed
                return var"#73#cres"[2]::Enzyme.eltype(var"#85#RetAnnotation")
            else
                if false
                    nothing
                else
                    Base.throw(Base.AssertionError("false"))
                end
            end
        else
            if var"#85#RetAnnotation" <: Enzyme.Const
                var"#73#cres" = Enzyme.ntuple(Enzyme.Val(var"#72#batchsize")) do var"#75#i"
                        Base.@_inline_meta
                        var"#76#dfn" = if var"#79#fn" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#79#fn").dval[var"#75#i"]
                            end
                        (ChainRulesCore).frule((var"#76#dfn", if var"#81#arg_1" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#81#arg_1").dval[var"#75#i"]
                                end, if var"#82#arg_2" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#82#arg_2").dval[var"#75#i"]
                                end, if var"#83#arg_3" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#83#arg_3").dval[var"#75#i"]
                                end), (var"#79#fn").val, (var"#81#arg_1").val, (var"#82#arg_2").val, (var"#83#arg_3").val; var"#84#kwargs"...)
                    end
                return (var"#73#cres"[1])[2]::Enzyme.eltype(var"#85#RetAnnotation")
            elseif #= /Users/pyong/.julia/packages/Enzyme/VSRgT/ext/EnzymeChainRulesCoreExt.jl:79 =# var"#85#RetAnnotation" <: Enzyme.BatchDuplicated
                var"#74#cres1" = begin
                        var"#75#i" = 1
                        var"#76#dfn" = if var"#79#fn" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#79#fn").dval[var"#75#i"]
                            end
                        (ChainRulesCore).frule((var"#76#dfn", if var"#81#arg_1" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#81#arg_1").dval[var"#75#i"]
                                end, if var"#82#arg_2" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#82#arg_2").dval[var"#75#i"]
                                end, if var"#83#arg_3" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#83#arg_3").dval[var"#75#i"]
                                end), (var"#79#fn").val, (var"#81#arg_1").val, (var"#82#arg_2").val, (var"#83#arg_3").val; var"#84#kwargs"...)
                    end
                var"#77#batches" = Enzyme.ntuple(Enzyme.Val(var"#72#batchsize" - 1)) do var"#94#j"
                        Base.@_inline_meta
                        var"#75#i" = var"#94#j" + 1
                        var"#76#dfn" = if var"#79#fn" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#79#fn").dval[var"#75#i"]
                            end
                        ((ChainRulesCore).frule((var"#76#dfn", if var"#81#arg_1" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#81#arg_1").dval[var"#75#i"]
                                end, if var"#82#arg_2" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#82#arg_2").dval[var"#75#i"]
                                end, if var"#83#arg_3" isa Enzyme.Const
                                    (ChainRulesCore).NoTangent()
                                else
                                    (var"#83#arg_3").dval[var"#75#i"]
                                end), (var"#79#fn").val, (var"#81#arg_1").val, (var"#82#arg_2").val, (var"#83#arg_3").val; var"#84#kwargs"...))[2]
                    end
                return Enzyme.BatchDuplicated(var"#74#cres1"[1], (var"#74#cres1"[2], var"#77#batches"...))
            elseif #= /Users/pyong/.julia/packages/Enzyme/VSRgT/ext/EnzymeChainRulesCoreExt.jl:92 =# var"#85#RetAnnotation" <: Enzyme.BatchDuplicatedNoNeed
                Enzyme.ntuple(Enzyme.Val(var"#72#batchsize")) do var"#75#i"
                    Base.@_inline_meta
                    var"#76#dfn" = if var"#79#fn" isa Enzyme.Const
                            (ChainRulesCore).NoTangent()
                        else
                            (var"#79#fn").dval[var"#75#i"]
                        end
                    ((ChainRulesCore).frule((var"#76#dfn", if var"#81#arg_1" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#81#arg_1").dval[var"#75#i"]
                            end, if var"#82#arg_2" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#82#arg_2").dval[var"#75#i"]
                            end, if var"#83#arg_3" isa Enzyme.Const
                                (ChainRulesCore).NoTangent()
                            else
                                (var"#83#arg_3").dval[var"#75#i"]
                            end), (var"#79#fn").val, (var"#81#arg_1").val, (var"#82#arg_2").val, (var"#83#arg_3").val; var"#84#kwargs"...))[2]
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
    @import_rrule typeof(find_alpha) Real Real Real
    @import_frule typeof(find_alpha) Real Real Real
end

end  # module
