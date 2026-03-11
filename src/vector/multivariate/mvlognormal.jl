struct MapLog end
(::MapLog)(x) = map(log, x)
function with_logabsdet_jacobian(::MapLog, x::AbstractArray{T}) where {T<:Number}
    y = map(log, x)
    return (y, -sum(y))
end
inverse(::MapLog) = MapExp()

struct MapExp end
(::MapExp)(x) = map(exp, x)
function with_logabsdet_jacobian(::MapExp, x::AbstractArray{T}) where {T<:Number}
    y = map(exp, x)
    return (y, sum(x))
end
inverse(::MapExp) = MapLog()

from_linked_vec(::D.AbstractMvLogNormal) = MapExp()
to_linked_vec(::D.AbstractMvLogNormal) = MapLog()
linked_vec_length(d::D.AbstractMvLogNormal) = length(d)
linked_optic_vec(d::D.AbstractMvLogNormal) = optic_vec(d)
