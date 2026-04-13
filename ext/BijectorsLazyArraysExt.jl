module BijectorsLazyArraysExt

import Bijectors: maporbroadcast
using LazyArrays: BroadcastArray

function maporbroadcast(f, x1::BroadcastArray, x...)
    return copy(f.(x1, x...))
end
function maporbroadcast(f, x1, x2::BroadcastArray, x...)
    return copy(f.(x1, x2, x...))
end
function maporbroadcast(f, x1, x2, x3::BroadcastArray, x...)
    return copy(f.(x1, x2, x3, x...))
end

end
