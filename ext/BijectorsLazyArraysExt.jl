module BijectorsLazyArraysExt

if isdefined(Base, :get_extension)
    using Bijectors: maporbroadcast
    using LazyArrays: LazyArrays
else
    using ..Bijectors: maporbroadcast
    using ..LazyArrays: LazyArrays
end

function maporbroadcast(f, x1::LazyArrays.BroadcastArray, x...)
    return copy(f.(x1, x...))
end
function maporbroadcast(f, x1, x2::LazyArrays.BroadcastArray, x...)
    return copy(f.(x1, x2, x...))
end
function maporbroadcast(f, x1, x2, x3::LazyArrays.BroadcastArray, x...)
    return copy(f.(x1, x2, x3, x...))
end

end
