const RTR = ReverseDiff.TrackedReal

_eps(::Type{<:RTR{T}}) where {T} = _eps(T)
