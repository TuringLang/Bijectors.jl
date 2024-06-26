module BijectorsTapirExt

if isdefined(Base, :get_extension)
    using Tapir: @from_rrule, MinimalCtx, Tapir
    using Bijectors: find_alpha
else
    using ..Tapir: @from_rrule, MinimalCtx, Tapir
    using ..Bijectors: find_alpha
end

@from_rrule MinimalCtx Tuple{typeof(find_alpha), Float64, Float64, Float64}

end
