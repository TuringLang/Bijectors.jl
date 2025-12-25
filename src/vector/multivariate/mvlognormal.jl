struct MapLog end
(::MapLog)(@nospecialize(x)) = map(log, x)
function with_logabsdet_jacobian(::MapLog, x::AbstractArray{T}) where {T<:Number}
    y = map(log, x)
    return (y, -sum(y))
end
inverse(::MapLog) = MapExp()

struct MapExp end
(::MapExp)(@nospecialize(x)) = map(exp, x)
function with_logabsdet_jacobian(::MapExp, x::AbstractArray{T}) where {T<:Number}
    y = map(exp, x)
    return (y, sum(x))
end
inverse(::MapExp) = MapLog()

from_linked_vec(::D.MvLogNormal) = MapExp()
to_linked_vec(::D.MvLogNormal) = MapLog()
linked_vec_length(d::D.MvLogNormal) = length(d)
linked_optic_vec(d::D.MvLogNormal) = optic_vec(d)
