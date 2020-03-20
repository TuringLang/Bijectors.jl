abstract type AbstractNamedBijector{names} <: AbstractBijector end

struct NamedBijector{names, Bs<:NamedTuple{names}} <: AbstractNamedBijector{names}
    bs::Bs
end

@inline names_to_bijectors(b::NamedBijector) = b.bs

@generated function (b::NamedBijector{names})(x::NamedTuple{names}) where {names}
    return :(($([:($n = b.bs.$n(x.$n)) for n in names]...), ))
end

@generated function Base.inv(b::NamedBijector{names}) where {names}
    return :(NamedBijector(($([:($n = inv(b.bs.$n)) for n in names]...), )))
end

@generated function logabsdetjac(b::NamedBijector{names}, x::NamedTuple{names}) where {names}
    return :(sum([$([:(logabsdetjac(b.bs.$n, x.$n)) for n in names]...), ]))
end

@generated function ∘(
        b1::NamedBijector{names}, 
        b2::NamedBijector{names}
) where {names}
    return :(NamedBijector(($([:($n = b1.bs.$n ∘ b2.bs.$n) for n in names]...), )))
end

# TODO: figure out a good way to do `composel` using generated functions?
function composel(bs::NamedBijector{names}...) where {names}
    return NamedBijector((; ((n, composel([b.bs[n] for b in bs]...)) for n in names)...))
end

function composer(bs::NamedBijector{names}...) where {names}
    return NamedBijector((; ((n, composer([b.bs[n] for b in bs]...)) for n in names)...))
end


####################
### NamedInverse ###
####################
struct NamedInverse{names, B<:AbstractNamedBijector{names}} <: AbstractNamedBijector{names}
    orig::B
end
Base.inv(nb::AbstractNamedBijector) = NamedInverse(nb)
Base.inv(ni::NamedInverse) = ni.orig

logabsdetjac(ni::NamedInverse, y::NamedTuple) = -logabsdetjac(inv(ni), ni(y))


############################
### `NamedCouplingLayer` ###
############################
struct NamedCoupling{names, target, deps, F} <: AbstractNamedBijector{names} where {F, target}
    f::F
end

NamedCoupling(names, target, deps) = NamedCoupling(names, target, deps, identity)
NamedCoupling(names, target, deps, f::F) where {F} = NamedCoupling{names, target, deps, F}(f)
function NamedCoupling(::Val{names}, ::Val{target}, ::Val{deps}, f::F) where {names, target, deps, F}
    return NamedCoupling{names, target, deps, F}(f)
end

Base.names(b::AbstractNamedBijector{names}) where {names} = names
coupling(b::NamedCoupling) = b.f
# For some reason trying to use the parameteric types doesn't always work
# so we have to do this weird approach of extracting type and then index `parameters`.
target(b::NamedCoupling) = typeof(b).parameters[2]
deps(b::NamedCoupling) = typeof(b).parameters[3]

@generated function (nc::NamedCoupling{names, target, deps, F})(x::NamedTuple) where {names, target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = b(x.$target), ))
    end
end

@generated function (ni::NamedInverse{names, <:NamedCoupling{names, target, deps, F}})(
    x::NamedTuple
) where {names, target, deps, F}
    return quote
        b = ni.orig.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = inv(b)(x.$target), ))
    end
end

@generated function logabsdetjac(nc::NamedCoupling{names, target, deps, F}, x::NamedTuple) where {names, target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return logabsdetjac(b, x.$target)
    end
end
