struct NamedBijector{Bs} <: AbstractBijector
    bs::Bs
end

@generated function (b::NamedBijector{Bs})(x::NamedTuple{names}) where {names, Bs<:NamedTuple{names}}
    return :(($([:($n = b.bs.$n(x.$n)) for n in names]...), ))
end

@generated function Base.inv(b::NamedBijector{Bs}) where {names, Bs<:NamedTuple{names}}
    return :(NamedBijector(($([:($n = inv(b.bs.$n)) for n in names]...), )))
end

@generated function logabsdetjac(b::NamedBijector{Bs}, x::NamedTuple{names}) where {names, Bs<:NamedTuple{names}}
    return :(sum([$([:(logabsdetjac(b.bs.$n, x.$n)) for n in names]...), ]))
end

@generated function ∘(
        b1::NamedBijector{<:NamedTuple{names}}, 
        b2::NamedBijector{<:NamedTuple{names}}
) where {names}
    return :(NamedBijector(($([:($n = b1.bs.$n ∘ b2.bs.$n) for n in names]...), )))
end

# TODO: figure out a good way to do `composel` using generated functions?
function composel(bs::NamedBijector{<:NamedTuple{names}}...) where {names}
    return NamedBijector((; ((n, composel([b.bs[n] for b in bs]...)) for n in names)...))
end

function composer(bs::NamedBijector{<:NamedTuple{names}}...) where {names}
    return NamedBijector((; ((n, composer([b.bs[n] for b in bs]...)) for n in names)...))
end
