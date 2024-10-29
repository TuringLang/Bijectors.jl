module BijectorsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme: @import_rrule, @import_frule
    using Bijectors: find_alpha
else
    using ..Enzyme: @import_rrule, @import_frule
    using ..Bijectors: find_alpha
end

@static if v"1.11.1" <= VERSION < v"1.12"
    @warn "Bijectors and Enzyme do not work together on Julia $VERSION"
else
    @import_rrule typeof(find_alpha) Real Real Real
    @import_frule typeof(find_alpha) Real Real Real
end

end  # module
