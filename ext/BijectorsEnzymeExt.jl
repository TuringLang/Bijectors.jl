module BijectorsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme: @import_frule, @import_rrule
    using Bijectors: find_alpha
else
    using ..Enzyme: @import_frule, @import_rrule
    using ..Bijectors: find_alpha
end

# Julia 1.11.1 caused a change in the ordering of precompilation for extensions.
# https://github.com/TuringLang/Bijectors.jl/issues/332
# See https://github.com/JuliaLang/julia/issues/56204
@static if v"1.11.1" <= VERSION < v"1.12"
    function __init__()
        if !Base.generating_output()
            eval(
                quote
                    @import_rrule typeof(find_alpha) Real Real Real
                    @import_frule typeof(find_alpha) Real Real Real
                end,
            )
        end
    end
else
    @import_rrule typeof(find_alpha) Real Real Real
    @import_frule typeof(find_alpha) Real Real Real
end

end
