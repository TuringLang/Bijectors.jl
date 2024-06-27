module BijectorsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme: @import_frule, @import_rrule
    using Bijectors: find_alpha
else
    using ..Tapir: @import_frule, @import_rrule
    using ..Bijectors: find_alpha
end

@import_rrule typeof(find_alpha) Float64 Float64 Float64
@import_frule typeof(find_alpha) Float64 Float64 Float64

end
